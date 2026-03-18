from __future__ import annotations
import os
import sys
import numpy as np
import torch
import warp as wp
import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM
import matplotlib.pyplot as plt
import noise
import random

from movement_primitives.dmp._dmp import DMP

""" ========== 沙子-刚体双向耦合：力计算内核 ========== """
@wp.kernel
def compute_body_forces(
    dt: float,# 时间步长。这是将“冲量”转换为“力”的关键参数 (F=I/t)。

    # 碰撞体ID数组。MPM求解器检测到与刚体有交互的网格节点ID。
    # 每个线程通过 wp.tid() 获得自己的索引 i，然后从 collider_ids[i] 中找到自己要处理的那个碰撞体ID cid。
    collider_ids: wp.array(dtype=int),

    # 碰撞冲量数组。与 collider_ids 对应，存储了每个碰撞体ID上受到的冲量大小和方向。
    collider_impulses: wp.array(dtype=wp.vec3),

    # 碰撞位置数组。与 collider_ids 对应，存储了每个碰撞体ID上冲量作用的位置（世界坐标系）。
    collider_impulse_pos: wp.array(dtype=wp.vec3),

    # 刚体ID映射表。
    # 这是一个“桥梁”，它将 MPM 求解器中的碰撞体ID (cid) 映射到刚体求解器中的刚体索引 (body_index)。
    # 如果一个碰撞体不属于任何刚体（比如地面），其对应的 body_index 可能为-1。
    body_ids: wp.array(dtype=int),

    # 刚体姿态数组。存储了所有刚体的位置和旋转（transform 包含一个向量 p 和一个四元数 q）。
    body_q: wp.array(dtype=wp.transform),

    # 刚体质心数组。存储了每个刚体在其局部坐标系下的质心位置。
    body_com: wp.array(dtype=wp.vec3),

    # 刚体空间力矢量数组 (输出)。这是函数要计算并填充的结果。
    # spatial_vector 是一个6维向量，前3维是 力 (force)，后3维是 力矩 (torque)。
    # 函数会把计算出的力和力矩原子地累加到这个数组中。
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """
    计算沙子施加在刚体上的力。
    将施加在每个MPM网格节点上的冲量相加, 并转换为主体质心处的力和扭矩。
    """

    # 1.获取当前线程处理的数据
    i = wp.tid()            # 当前线程索引
    cid = collider_ids[i]   # 根据线程ID从输入数组中取得该线程需要处理的碰撞体ID

    # 2.有效性检查
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid] # 查询这个碰撞体属于哪个刚体
        if body_index == -1:       # -1意味着这个碰撞发生在非刚体上（例如固定的地面），无需计算直接返回
            return

        # 3.将冲量转换为力 (F=I/t)
        f_world = collider_impulses[i] / dt 

        # 4.计算力矩 (Torque)
        X_wb = body_q[body_index]    # 刚体的世界变换（位置和姿态）。
        X_com = body_com[body_index] # 局部坐标系下的质心位置。
        # 力的作用点在世界坐标系中的位置与刚体质心在世界坐标系中的位置之差，得到力矩臂 r。
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)

        # 5.累加力和力矩，原子操作以避免数据竞争
        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


""" ========== 沙子-刚体双向耦合：力减去内核 ========== """
@wp.kernel
def subtract_body_force(
    dt: float, # 时间步长。用于计算速度变化量。
    body_q: wp.array(dtype=wp.transform),           # [输入] 刚体当前的姿态（位置和旋转）。
    body_qd: wp.array(dtype=wp.spatial_vector),     # [输入] 刚体当前的速度（线速度和角速度）。这就是我们上面说的 v_final。
    body_f: wp.array(dtype=wp.spatial_vector),      # [输入] 在刚体步中施加的、来自沙子的力 F_sand。
    body_inv_inertia: wp.array(dtype=wp.mat33),     # [输入] 刚体的 逆惯量张量 (在局部坐标系下)。
    body_inv_mass: wp.array(dtype=float),           # [输入] 刚体的 逆质量 (1/mass)。
    body_q_res: wp.array(dtype=wp.transform),       # [输出] 结果姿态。这里直接复制输入姿态，因为我们只修改速度。
    body_qd_res: wp.array(dtype=wp.spatial_vector), # [输出] 结果速度。这就是计算出的“干净”速度 v_base。
):
    """
    更新刚体速度，以移除上一步沙子施加的力。
    这是计算执行基于互补性的摩擦接触边界条件所需的总冲量所必需的。
    """

    # 1.获取当前处理的刚体
    body_id = wp.tid()

    # 2.计算线速度变化量 (Δv)
    f = body_f[body_id] # 获取施加在该刚体上的空间力矢量
    # wp.spatial_top(f)：从6维空间力矢量中提取前3维，即 线性力 F。
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f) # 计算线速度变化量 Δv = (F/m) * dt

    # 3.计算角速度变化量 (Δω)
    r = wp.transform_get_rotation(body_q[body_id])
    # wp.spatial_bottom(f): 从空间力矢量中提取后3维，即 力矩 τ (在世界坐标系下)。
    # τ = I * α = I * Δω / Δt =====> Δω = I⁻¹ * τ * Δt
    # 坐标系变换:
    # wp.quat_rotate_inv(r, ...): 使用刚体旋转 r 的逆，将世界坐标系下的力矩 τ 变换到 刚体的局部坐标系。
    # body_inv_inertia[body_id] * ...: 在局部坐标系下，安全地乘以逆惯量张量，得到局部坐标系下的角加速度。
    # wp.quat_rotate(r, ...): 再使用刚体旋转 r，将计算出的局部角加速度变换回 世界坐标系。
    # delta_w = dt * ...: 最后乘以时间步长 dt，得到角速度的变化量。
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    # 4.更新结果姿态和速度
    # 姿态 body_q 不变，直接复制过去。
    # body_qd[body_id] 是刚体的最终速度 v_final。
    # wp.spatial_vector(delta_v, delta_w) 是由沙子力产生的总速度变化。
    # 两者相减，就得到了我们想要的“干净”基准速度 v_base，并存入输出数组 body_qd_res。
    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


""" ========== 沙子-刚体交互：统计单个 body 的接触力 ========== """
@wp.kernel
def compute_single_body_contact_force(
    dt: float,
    target_body: int,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    out_force: wp.array(dtype=wp.vec3),
):
    """把 MPM collider 冲量累加为某个刚体的接触力（世界系）。

    注意：这是“检测/统计”用途，不影响刚体求解。
    """

    i = wp.tid()
    cid = collider_ids[i]
    if cid < 0 or cid >= body_ids.shape[0]:
        return

    body_index = body_ids[cid]
    if body_index != target_body:
        return

    f_world = collider_impulses[i] / dt
    wp.atomic_add(out_force, 0, f_world)


""" ========== 机器人-沙子交互：动态碰撞体更新内核 ========== """
@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),                # 机器人腿部网格的静止顶点（形状局部坐标）
    src_shape: wp.array(dtype=int),                     # 每个顶点对应的形状ID
    res_mesh: wp.uint64,                                # 目标碰撞网格（用于沙子交互）
    shape_transforms: wp.array(dtype=wp.transform),     # 形状到刚体的变换
    shape_body_id: wp.array(dtype=int),                 # 形状所属的刚体ID
    body_q: wp.array(dtype=wp.transform),               # 刚体的世界变换（机器人各部件姿态）
    dt: float,                                          # 时间步长
):
    """
    功能：更新碰撞网格，用于与沙子的物理交互
    过程：静止顶点 → 形状坐标 → 刚体坐标 → 世界坐标，并计算顶点速度
    """
    v = wp.tid()  # 当前顶点索引

    # 添加边界检查，防止数组越界
    if v >= src_points.shape[0]:
        return

    res = wp.mesh_get(res_mesh)  # 获取碰撞网格

    # 获取顶点所属的形状和刚体
    shape_id = src_shape[v]
    # 将静止顶点从形状局部坐标转换到刚体局部坐标
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    # 获取刚体的世界变换（机器人腿部的当前姿态）
    X_wb = body_q[shape_body_id[shape_id]]

    # 计算顶点在世界坐标系下的新位置
    cur_p = res.points[v] + dt * res.velocities[v]  # 当前预测位置
    next_p = wp.transform_point(X_wb, p)            # 实际新位置

    # 更新顶点速度和位置（用于沙子碰撞检测）
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p

class Example:
    # === [NEW] DMP CONTROL (2D) BEGIN =====================================
    class _DMPCouplingTerm:
        """DMP 耦合项(coupling term)。

        作用：给 DMP 的加速度方程额外叠加一个“外力/耦合”项，用来实现：
        - `force_eta`: 类似 action 的连续控制输入（这里直接作为加速度偏置）。
        - `finish_flag`: 终止/冻结某些维度（例如触地后不再继续“下压”）。

        说明:movement_primitives 的 DMP.step() 会在内部调用 coupling_term.coupling(y, yd)
        来获取 (cd, cdd)。其中：
        - cd: 速度层面的耦合项（这里取 0)。
        - cdd: 加速度层面的耦合项（这里用 force_eta, 并按 finish_flag 置零）。
        """
        def __init__(self, n_dims: int):
            # n_dims：DMP 维度数。此例为 2（hip/knee）。
            self.n_dims = int(n_dims)
            # force_eta：外力项（每维一个），类型 float64，便于和 DMP 内部 numpy 计算对齐。
            self.force_eta = np.zeros(self.n_dims, dtype=np.float64)
            # finish_flag：停止标志（每维一个），1 表示该维度“停止施加外力/耦合”。
            self.finish_flag = np.zeros(self.n_dims, dtype=np.int32)

        def coupling(self, y, yd):
            """返回 (cd, cdd)。

            Args:
                y: 当前 DMP 位置（此处未使用，仅为接口要求）。
                yd: 当前 DMP 速度（此处未使用，仅为接口要求）。

            Returns:
                cd: 速度耦合项(n_dims,），此处恒为 0。
                cdd: 加速度耦合项(n_dims,）。默认等于 force_eta;若 finish_flag 为 1 则置 0。
            """
            # 1) 把外力/标志转换成固定形状的一维向量，避免 dtype/shape 传播到 DMP 内部导致错误。
            force = np.asarray(self.force_eta, dtype=np.float64).reshape(self.n_dims)
            finish = np.asarray(self.finish_flag, dtype=np.int32).reshape(self.n_dims)

            # 2) 加速度耦合项：直接使用 force_eta 作为“加速度偏置”。
            #    注意：这不是物理意义上的力（未乘质量/惯量），而是 DMP 动力学方程的额外项。
            cdd = force.copy()

            # 3) 若某一维 finish_flag 为 True：认为该维度已经“完成/触发停止条件”，
            #    则把该维度耦合项置零，避免继续往目标方向推。
            cdd[finish.astype(bool)] = 0.0

            # 4) 速度耦合项这里不使用，返回全零。
            cd = np.zeros_like(cdd)
            return cd, cdd

    # === [NEW] DMP CONTROL (2D) END =======================================

    def __init__(self, viewer, options):
        # ========== 仿真时间控制 ==========
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps        # 每帧时间步长
        
        self.sim_time = 0.0                   # 当前仿真时间
        self.sim_substeps = options.substeps  # 每帧的子步数
        self.sim_dt = self.frame_dt / self.sim_substeps  # 每个子步的时间步长

        self.viewer = viewer
        self._headless_null_viewer = self.viewer.__class__.__name__.lower().endswith("null")
        self._contact_seen_frames = 0

        # 控制模式：
        # - dmp: 关节空间 DMP（当前默认实现）
        # - script: 手动脚本轨迹（抬腿->下压->保持），便于手动调参/验证
        self.leg_controller = str(getattr(options, "leg_controller", "dmp")).lower()
        if self.leg_controller not in ("dmp", "script"):
            raise ValueError("--leg-controller must be one of: dmp, script")

        # 估算地形表面高度：粒子发射区域上界 + 轻微裕量（近似粒子半径/表面起伏）
        emit_hi = getattr(options, "emit_hi", [0.5, 0.5, 0.25])
        voxel_size = float(getattr(options, "voxel_size", 0.03))
        self.terrain_z_top = float(emit_hi[2]) + 0.5 * voxel_size

        # ========== 沙子-物块交互系统构建 ==========
        builder = newton.ModelBuilder()

        # 关节/碰撞默认参数（URDF 导入的机器人也会复用这些默认值）
        builder.default_joint_cfg.target_ke = float(getattr(options, "joint_target_ke", 2000.0))
        builder.default_joint_cfg.target_kd = float(getattr(options, "joint_target_kd", 50.0))
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        friction_coeff = float(getattr(options, "friction_coeff", 0.48))
        builder.default_shape_cfg.mu = friction_coeff

        # 交互相关：添加地面作为沙子的底部边界
        ground_shape = builder.add_ground_plane()

        # 只把 ground plane 用作 MPM 粒子边界：避免机器人/刚体与该平面发生刚体接触
        # （机器人应只与 MPM 粒子地形交互；否则会出现“踩在刚体地面上”而非踩粒子的情况。）
        if hasattr(builder, "shape_flags"):
            builder.shape_flags[ground_shape] = builder.shape_flags[ground_shape] & ~newton.ShapeFlags.COLLIDE_SHAPES

        # ========== 机器人模型导入（URDF） ==========
        # 默认使用同目录下的 el_mini URDF（你刚拖进来的文件夹）
        _this_dir = os.path.dirname(__file__)
        _default_urdf = os.path.join(_this_dir, "el_mini", "urdf", "el_mini.urdf")
        robot_urdf = str(getattr(options, "robot_urdf", _default_urdf))

        robot_floating = bool(getattr(options, "robot_floating", False))
        base_z_in = float(getattr(options, "robot_base_z", 0.3))

        # 自动修正 base_z：用一次临时导入估算当前足端高度，然后把足端移到地形表面上方一点
        base_z = base_z_in
        # if not robot_floating:
        #     try:
        #         base_z = self._auto_base_z_for_contact(
        #             robot_urdf=robot_urdf,
        #             base_z=base_z_in,
        #             foot_link=str(getattr(options, "foot_link", "LF_SHANK")),
        #             terrain_z_top=self.terrain_z_top,
        #         )
        #     except Exception:
        #         base_z = base_z_in

        base_pos = wp.vec3(0.0, 0.0, float(base_z))
        base_rot = wp.quat_identity()

        builder.add_urdf(
            robot_urdf,
            xform=wp.transform(p=base_pos, q=base_rot),
            floating=robot_floating,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # 只让“六条腿足端”几何参与 MPM 粒子碰撞：
        # - 避免整机 mesh 全部参与导致很慢
        # - 避免小腿/电机等部件也“钻地”
        #
        # 规则：优先用 shape_key 中包含 FOOT/TOE/SOLE 的 shapes 作为足端；
        # 若没有找到，则退化为每条腿的 FOOT(或 SHANK) body 的全部 shapes。
        foot_link = str(getattr(options, "foot_link", "LF_SHANK"))
        if hasattr(builder, "body_key") and hasattr(builder, "body_shapes") and hasattr(builder, "shape_flags"):
            body_names = [str(k) for k in list(builder.body_key)]

            # 1) 兜底选择一个“主足端 body”（用于后续取力/触地判定等逻辑，默认仍以 LF 为主）。
            if foot_link not in builder.body_key:
                foot_link_found = next((k for k in body_names if "FOOT" in k.upper()), None)
                if foot_link_found is None:
                    foot_link_found = next((k for k in body_names if "SHANK" in k.upper()), None)
                if foot_link_found is None and body_names:
                    foot_link_found = body_names[-1]
                if foot_link_found is not None:
                    foot_link = foot_link_found

            # 2) 自动挑出六条腿的“足端 bodies”。
            #    约定：腿前缀为 LF/RF/LM/RM/LB/RB（若你的 URDF 不同，可后续再加一个显式列表参数）。
            leg_prefixes = ("LF", "RF", "LM", "RM", "LB", "RB")
            foot_bodies: list[int] = []

            def _pick_body(prefix: str, tokens: tuple[str, ...]) -> int | None:
                prefix_u = prefix.upper()
                for name in body_names:
                    up = name.upper()
                    if up.startswith(prefix_u) and any(t in up for t in tokens):
                        return int(builder.body_key.index(name))
                return None

            for prefix in leg_prefixes:
                idx = _pick_body(prefix, ("FOOT", "TOE", "SOLE"))
                if idx is None:
                    idx = _pick_body(prefix, ("SHANK",))
                if idx is not None:
                    foot_bodies.append(int(idx))

            # 若没匹配到六足前缀：退化为“所有名字里包含 FOOT/TOE/SOLE 的 body”。
            if not foot_bodies:
                for i, name in enumerate(body_names):
                    up = name.upper()
                    if ("FOOT" in up) or ("TOE" in up) or ("SOLE" in up):
                        foot_bodies.append(int(i))

            # 最后兜底：至少保留主 foot_link 所在的 body。
            if (not foot_bodies) and (foot_link in builder.body_key):
                foot_bodies = [int(builder.body_key.index(foot_link))]

            foot_bodies = sorted(set(int(b) for b in foot_bodies))

            # 3) 收集“允许与粒子碰撞”的 shapes：六足足端 shapes + ground plane。
            keep_shape_ids: list[int] = []
            if hasattr(builder, "shape_key"):
                for body in foot_bodies:
                    for sid in builder.body_shapes[body]:
                        key = str(builder.shape_key[sid]).upper()
                        if ("FOOT" in key) or ("TOE" in key) or ("SOLE" in key):
                            keep_shape_ids.append(int(sid))

            # 若 shape_key 没有足端标记：退化为保留足端 bodies 的全部 shapes。
            if not keep_shape_ids:
                for body in foot_bodies:
                    keep_shape_ids.extend(int(s) for s in builder.body_shapes[body])

            keep_shape_set = set(int(s) for s in keep_shape_ids)
            keep_shape_set.add(int(ground_shape))

            # 4) 对除了 keep_shape_set 之外的所有 shapes 关闭粒子碰撞。
            for body in range(builder.body_count):
                for shape in builder.body_shapes[body]:
                    if int(shape) not in keep_shape_set:
                        builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        # 记录足端 body 索引，用于后续取力/判定触地
        self.foot = 0
        if hasattr(builder, "body_key") and foot_link in builder.body_key:
            self.foot = int(builder.body_key.index(foot_link))

        # 记录机器人基座 body 索引（用于构造“足端相对位姿”等观测）
        # 约定：el_mini URDF 的基座 link 名为 "BASE"。
        self.base = 0
        if hasattr(builder, "body_key"):
            if "BASE" in builder.body_key:
                self.base = int(builder.body_key.index("BASE"))
            else:
                # 兜底：优先挑包含 BASE/TRUNK/TORSO/ROOT 的 body key
                keys = [str(k).upper() for k in list(builder.body_key)]
                for token in ("BASE", "TRUNK", "TORSO", "ROOT"):
                    idx = next((i for i, k in enumerate(keys) if token in k), None)
                    if idx is not None:
                        self.base = int(idx)
                        break

        # 选择 DMP 控制的关节（默认只控 LF 腿三关节：HAA/HFE/KFE）。
        # 设计目标：导入整机，但只控 LF 腿，其余关节保持初始目标（等价“锁定”）。
        dmp_joints = getattr(options, "dmp_joints", None)
        if dmp_joints is None:
            dmp_joints = ["LF_HAA", "LF_HFE", "LF_KFE"]
        self.dmp_joint_names = [str(x) for x in list(dmp_joints)]
        dof_offset = 6 if robot_floating else 0
        self.dmp_dof_indices: list[int] = []
        if hasattr(builder, "joint_key"):
            for name in self.dmp_joint_names:
                if name not in builder.joint_key:
                    raise ValueError(f"DMP joint '{name}' not found in URDF joints")
                self.dmp_dof_indices.append(int(builder.joint_key.index(name) + dof_offset))

        self.dmp_n_dims = int(len(self.dmp_dof_indices))
        if self.dmp_n_dims < 1:
            raise ValueError("dmp_joints must contain at least 1 joint")
        
        # ========== 沙子模型构建 ==========
        mpm_builder = newton.ModelBuilder()
        # 沙子相关：在指定区域生成 MPM 粒子
        sand_particles, snow_particles, mud_particles, block_particles = Example.emit_particles(mpm_builder, options)

        # ========== 物理模型最终化 ==========
        self.model = builder.finalize()
        self.mpm_model = mpm_builder.finalize()

        # 足端“与粒子发生碰撞”的 shape 顶点（已变换到 body frame），用于计算足底最低点高度
        self._foot_collision_points_body = self._build_body_collision_points_body_frame(self.model, self.foot)

        # 沙子相关：设置沙子的摩擦属性
        self.mpm_model.particle_mu = friction_coeff
        self.mpm_model.particle_kd = 0.0
        self.mpm_model.particle_ke = 1.0e15

        sand_particles = wp.array(sand_particles, dtype=int, device=self.mpm_model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=self.mpm_model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=self.mpm_model.device)
        block_particles = wp.array(block_particles, dtype=int, device=self.mpm_model.device)

        # 沙子相关：设置重力（影响沙子下落和堆积）
        gravity = getattr(options, "gravity", [0.0, 0.0, -10.0])
        self.model.set_gravity(gravity)
        self.mpm_model.set_gravity(gravity)

        # ========== 沙子 MPM 求解器配置 ==========
        # 沙子相关：创建 MPM 求解器选项并复制命令行参数
        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # CPU 设备上，sparse grid 的 allocate_by_tiles 路径仅支持 CUDA。
        # 当外部调用（例如 RL env）未显式设置 --grid-type 时，这里做一次安全回退。
        if hasattr(mpm_options, "grid_type") and (not wp.get_device().is_cuda):
            if getattr(mpm_options, "grid_type", None) == "sparse":
                mpm_options.grid_type = "fixed"

        # 沙子相关：创建 MPM 模型（将 Newton 模型适配为 MPM 格式）
        mpm_model = SolverImplicitMPM.Model(self.mpm_model, mpm_options)

        # --- 雪的属性 (类雪材料) ---
        mpm_model.material_parameters.yield_pressure[snow_particles].fill_(2.0e4)  # 屈服压力 (较低，易于压缩)
        mpm_model.material_parameters.yield_stress[snow_particles].fill_(1.0e3)  # 屈服应力 (较低，易于剪切变形)
        mpm_model.material_parameters.tensile_yield_ratio[snow_particles].fill_(0.05)  # 拉伸屈服比 (抗拉伸能力弱)
        mpm_model.material_parameters.friction[snow_particles].fill_(0.1)  # 内摩擦角 (较低)
        mpm_model.material_parameters.hardening[snow_particles].fill_(10.0)  # 硬化系数 (压缩后会变硬)

        # --- 泥浆的属性 (类流体材料) ---
        mpm_model.material_parameters.yield_pressure[mud_particles].fill_(1.0e10)  # 屈服压力 (非常高，几乎不可压缩)
        mpm_model.material_parameters.yield_stress[mud_particles].fill_(3.0e2)  # 屈服应力 (非常低，易于流动)
        mpm_model.material_parameters.tensile_yield_ratio[mud_particles].fill_(1.0)  # 拉伸屈服比 (具有一定的抗拉伸能力)
        mpm_model.material_parameters.hardening[mud_particles].fill_(2.0)  # 硬化系数 (较低)
        mpm_model.material_parameters.friction[mud_particles].fill_(0.0)  # 内摩擦角 (为0，表现更像流体)

        # --- 石块的属性 (固体材料) ---
        mpm_model.material_parameters.yield_pressure[block_particles].fill_(1.0e8)  # 屈服压力 (较高，难以压缩)
        mpm_model.material_parameters.yield_stress[block_particles].fill_(1.0e7)  # 屈服应力 (非常高，抵抗剪切变形能力强)
        mpm_model.material_parameters.tensile_yield_ratio[block_particles].fill_(0.8)  # 拉伸屈服比 (抗拉伸能力强)
        mpm_model.material_parameters.friction[block_particles].fill_(0.6)  # 内摩擦角 (高，颗粒间滑动困难)
        mpm_model.material_parameters.hardening[block_particles].fill_(20.0)  # 硬化系数 (高，压缩后会变得更硬)

        # 通知MPM模型，粒子材质参数已发生变化
        mpm_model.notify_particle_material_changed()

        # 从刚体模型而不是沙子模型中读取碰撞体
        mpm_model.setup_collider(model=self.model)

        # ========== 求解器初始化 ==========
        # 物块求解器
        # 增加求解器迭代次数以提高稳定性，尤其是在处理复杂关节和外部交互时。
        # 更多的迭代可以让XPBD更好地解算约束。
        self.solver_block = newton.solvers.SolverXPBD(self.model, iterations=10)

        # 沙子 MPM 求解器
        self.solver_mpm = SolverImplicitMPM(mpm_model, mpm_options)

        # ========== 单腿系统状态初始化 ==========
        # === [NEW] DMP CONTROL (2D) BEGIN =====================================
        # 说明：该腿的两个转动关节 axis 都是 Y，因此主要在 XZ 平面摆动。
        # 这里实现一个“2 自由度关节空间 DMP”，并额外加入：
        # - 外力项：force_eta = clip(action)*scale（把连续 action 映射到 DMP 加速度偏置）。
        # - 停止条件：finish_flag（例如检测到触地法向力/切向力超过阈值，则停止继续施加外力）。
        #
        # 注意：本文件中 DMP 的状态 (pos/vel) 在 CPU 侧（numpy）更新，
        #       每个子步将 joint_target_pos/vel 拷贝到 device（warp）供 XPBD 使用。
        #       这也是为什么在 capture() 里禁用了 CUDA Graph capture（host<->device 交换）。
        self.dmp_action_bound = (-1.0, 1.0)
        # action -> force_eta 的比例系数。数值越大，DMP 被“外力项”推得越狠（更像主动控制）。
        self.dmp_force_scale = float(getattr(options, "dmp_force_scale", 500.0))
        # finish_flag 阈值：用于判定“触地/摩擦”是否达到停止外力的条件。
        # - dmp_fn_max：法向力阈值（这里从足端空间力的 z 分量估计）。
        # - dmp_ft_max：切向力阈值（这里用 x/y 的合力估计）。
        self.dmp_fn_max = float(getattr(options, "dmp_fn_max", 400.0))
        self.dmp_ft_max = float(getattr(options, "dmp_ft_max", 400.0))
        # dmp_action：N 维输入（默认 3：LF_HAA/LF_HFE/LF_KFE），对应每个关节的外力耦合项。
        default_action = [0.0] * self.dmp_n_dims
        self.dmp_action = np.array(getattr(options, "dmp_action", default_action), dtype=np.float32)
        if int(self.dmp_action.shape[0]) != self.dmp_n_dims:
            # 允许用户仍传 2 维：则自动补 0（常用于只控 HFE/KFE）。
            a = np.zeros(self.dmp_n_dims, dtype=np.float32)
            n_copy = min(self.dmp_n_dims, int(self.dmp_action.shape[0]))
            a[:n_copy] = self.dmp_action[:n_copy]
            self.dmp_action = a

        # finish_flag：N 维停止标志，后续在 simulate_leg() 里根据足端力更新。
        self.finish_flag = np.zeros(self.dmp_n_dims, dtype=np.int32)
        # === [NEW] DMP CONTROL (2D) END =======================================

        # ========== 仿真状态初始化 ==========        
        # 沙子相关：创建双缓冲状态（用于时间步进）
        self.state_0 = self.model.state()  # 当前状态
        self.state_1 = self.model.state()  # 下一状态

        self.sand_state_0 = self.mpm_model.state()
        # 沙子相关：为状态添加 MPM 专用字段（网格信息、粒子缓存等）
        self.solver_mpm.enrich_state(self.sand_state_0)

        # 初始化机器人正向运动学和碰撞网格
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # === [NEW] DMP CONTROL (2D) BEGIN =====================================
        # 控制对象：用于 XPBD 的关节目标。
        # XPBD 在 step() 时，会读取：
        # - control.joint_target_pos：目标关节角（rad）
        # - control.joint_target_vel：目标关节角速度（rad/s）
        # 并结合 model.joint_target_ke/kd（类似 PD 增益）生成约束/驱动力。
        self.control = self.model.control()

        # 给关节目标加一点“PD 增益”。
        # 一些模型默认 ke/kd 可能为 0：即便你写入了 target_pos/vel，也不会产生收敛驱动力，
        # 现象就是“目标写了但关节不动/会复位”。
        if self.model.joint_target_ke is not None and self.model.joint_target_kd is not None:
            ke = float(getattr(options, "joint_target_ke", 2000.0))
            kd = float(getattr(options, "joint_target_kd", 50.0))
            self.model.joint_target_ke.assign([ke] * int(self.model.joint_dof_count))
            self.model.joint_target_kd.assign([kd] * int(self.model.joint_dof_count))

        # 初始化二维 DMP（关节空间：[hip, knee]，单位 rad）。
        # 这里用当前 state_0.joint_q 作为 DMP 的起点（start_y），再由 _init_dmp_joint_space()
        # 构造一条参考轨迹并做 imitate()，得到可泛化的 DMP。
        #
        # 注意：initial_q 取自 state_0.joint_q 的 numpy 拷贝（host 侧），避免直接在 device 上读写。
        initial_q_full = self.state_0.joint_q.numpy().astype(np.float32)
        self._dmp_ref_joint_q = initial_q_full.astype(np.float64)
        start_y = initial_q_full[self.dmp_dof_indices].astype(np.float32)

        # host 侧控制（DMP 或脚本）需要每子步在 CPU 侧更新目标，并写入 device；不适配 CUDA Graph capture。
        self._disable_cuda_graph = True

        if self.leg_controller == "dmp":
            self.dmp = self._init_dmp_joint_space(options, start_y)
            # DMP 内部使用 float64 计算更稳（movement_primitives 也常用 float64）。
            self.dmp_pos = self.dmp.start_y.astype(np.float64)
            self.dmp_vel = np.zeros_like(self.dmp_pos, dtype=np.float64)
            # 耦合项对象：用于把 action/停止信号注入到 DMP 的 step() 中。
            self._dmp_coupling = self._DMPCouplingTerm(n_dims=self.dmp_n_dims)
        else:
            # script 模式：不创建 DMP；由 simulate_leg() 生成分段轨迹并写入关节目标。
            self.dmp_pos = np.asarray(start_y, dtype=np.float64).reshape(self.dmp_n_dims)
            self.dmp_vel = np.zeros_like(self.dmp_pos, dtype=np.float64)
            self._dmp_coupling = None
            self._script_t0 = float(self.sim_time)
            self._script_prev_target = self.dmp_pos.copy()

            # 脚本参数（可通过命令行覆盖）
            self.script_cycle_time = float(getattr(options, "script_cycle_time", 1.2))
            self.script_lift_ratio = float(getattr(options, "script_lift_ratio", 0.25))
            self.script_press_ratio = float(getattr(options, "script_press_ratio", 0.65))
            self.script_stop_on_contact = bool(getattr(options, "script_stop_on_contact", True))
            self.script_exp_beta = float(getattr(options, "script_exp_beta", 2.0))
            self.script_press_scale = float(getattr(options, "script_press_scale", 1.0))
            self.script_lift_scale = float(getattr(options, "script_lift_scale", 0.6))

            # 复用 DMP 的“每关节下压幅度”覆盖逻辑
            default_press = {
                "LF_HAA": 0.25,
                "LF_HFE": 0.4,
                "LF_KFE": 0.8,
            }
            base_delta = np.zeros(self.dmp_n_dims, dtype=np.float64)
            for i, name in enumerate(self.dmp_joint_names):
                key = str(name)
                opt_key = f"dmp_press_{key.lower()}"
                if hasattr(options, opt_key):
                    base_delta[i] = float(getattr(options, opt_key))
                else:
                    base_delta[i] = float(default_press.get(key, 0.4))
            self._script_base_delta = base_delta

            # 可选：用户显式给出逐关节的 lift/press 幅度（rad）。若未给出，则用 base_delta*scale。
            press_delta_in = getattr(options, "script_press_delta", None)
            lift_delta_in = getattr(options, "script_lift_delta", None)
            if press_delta_in is not None:
                press_delta = np.asarray(list(press_delta_in), dtype=np.float64).reshape(-1)
                if press_delta.shape[0] != self.dmp_n_dims:
                    tmp = np.zeros(self.dmp_n_dims, dtype=np.float64)
                    n_copy = min(self.dmp_n_dims, int(press_delta.shape[0]))
                    tmp[:n_copy] = press_delta[:n_copy]
                    press_delta = tmp
            else:
                press_delta = base_delta * self.script_press_scale

            if lift_delta_in is not None:
                lift_delta = np.asarray(list(lift_delta_in), dtype=np.float64).reshape(-1)
                if lift_delta.shape[0] != self.dmp_n_dims:
                    tmp = np.zeros(self.dmp_n_dims, dtype=np.float64)
                    n_copy = min(self.dmp_n_dims, int(lift_delta.shape[0]))
                    tmp[:n_copy] = lift_delta[:n_copy]
                    lift_delta = tmp
            else:
                lift_delta = base_delta * self.script_lift_scale

            # 计算“下压方向” sign：沿着让足端最低点 z 降低的方向
            def _foot_min_z_from_joint(q: np.ndarray) -> float:
                tmp_state = self.model.state()
                q_full = np.array(self._dmp_ref_joint_q, dtype=np.float64, copy=True)
                q_full[self.dmp_dof_indices] = np.asarray(q, dtype=np.float64).reshape(self.dmp_n_dims)
                tmp_state.joint_q.assign(q_full.tolist())
                if tmp_state.joint_qd is not None:
                    qd_len = int(tmp_state.joint_qd.shape[0])
                    if qd_len > 0:
                        tmp_state.joint_qd.assign([0.0] * qd_len)
                newton.eval_fk(self.model, tmp_state.joint_q, tmp_state.joint_qd, tmp_state)
                return float(self._foot_min_z_from_state(tmp_state))

            start = np.asarray(self.dmp_pos, dtype=np.float64).reshape(self.dmp_n_dims)
            eps = 1e-3
            sign = np.ones(self.dmp_n_dims, dtype=np.float64)
            for j in range(self.dmp_n_dims):
                dq = np.zeros(self.dmp_n_dims, dtype=np.float64)
                dq[j] = eps
                z_plus = _foot_min_z_from_joint(start + dq)
                z_minus = _foot_min_z_from_joint(start - dq)
                sign[j] = 1.0 if (z_plus < z_minus) else -1.0
            self._script_sign = sign

            # 固定的三段轨迹关键点（以启动时关节角为基准）
            self._script_start = start.copy()
            self._script_press_goal = start + sign * press_delta
            self._script_lift_goal = start - sign * lift_delta

        # 初始把 XPBD 目标设置为当前关节角/速度。
        # 目的：避免第一帧就从默认目标（可能为 0）瞬间拉到当前姿态，引入不必要的冲击。
        _target_pos = initial_q_full.astype(np.float64)
        _target_vel = np.zeros_like(_target_pos)
        _target_pos[self.dmp_dof_indices] = self.dmp_pos.reshape(self.dmp_n_dims)
        _target_vel[self.dmp_dof_indices] = self.dmp_vel.reshape(self.dmp_n_dims)
        self.control.joint_target_pos.assign(_target_pos.tolist())
        self.control.joint_target_vel.assign(_target_vel.tolist())

        # 用于“下一子步”更新 finish_flag 的上一时刻足端力。
        # 注意：我们是在本子步中先根据上一子步的力更新 finish_flag，再执行 DMP.step()。
        # 这样可以把“触地”信息反馈到下一次 DMP 更新（避免同一子步内出现循环依赖）。
        self._last_foot_force_world = np.zeros(3, dtype=np.float32)
        # === [NEW] DMP CONTROL (2D) END =======================================

        # 刚体相关：初始化碰撞检测
        self.contacts = self.model.collide(self.state_0)

        # ========== 双向耦合数据缓冲区 ==========
        # 用于跟踪双向耦合力的附加缓冲区
        # 按位左移 (Bitwise Left Shift) 运算符。
        # 它会将左侧操作数（这里是1）的所有二进制位向左移动指定的位数（右侧操作数，这里是20）。右侧空出的位用0填充。
        max_nodes = 1 << 20  # 2的20次方，约为1048576
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self._n_colliders = 0
        self.collect_collider_impulses()

        # 从碰撞体索引到刚体索引的映射
        self.collider_body_id = mpm_model.collider.collider_body_index

        # 沙子施加到刚体上的每个刚体的力和扭矩
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        # 足端与 MPM 粒子的交互力（检测用，世界系），在 simulate_mpm() 后更新
        self.foot_particle_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.foot_particle_force_world = np.zeros(3, dtype=np.float32)

        self.particle_render_colors = wp.full(
            self.mpm_model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.mpm_model.device
        )

        self.particle_render_colors[sand_particles].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_render_colors[snow_particles].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_render_colors[mud_particles].fill_(wp.vec3(0.4, 0.25, 0.25))
        self.particle_render_colors[block_particles].fill_(wp.vec3(0.5, 0.5, 0.5))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True  # 显示沙子粒子
        self.show_impulses = True

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # ========== F-t 曲线绘制设置 ==========
        self.force_history = []
        self.time_history = []
        self.enable_plot = bool(getattr(options, "enable_plot", True))
        if self.enable_plot:
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot(self.time_history, self.force_history)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Force (N)")
            self.ax.set_title("Interaction Force vs. Time")
            self.ax.grid(True)
        else:
            self.fig = None
            self.ax = None
            self.line = None
        self.capture()

    def capture(self):
        """设置CUDA图优化以加速仿真"""
        self.graph = None
        # host 侧控制（DMP/脚本）与 host<->device 数据交换不适配 CUDA Graph capture
        if bool(getattr(self, "_disable_cuda_graph", False)):
            self.sand_graph = None
            return
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_leg()
            self.graph = capture.graph
        
        self.sand_graph = None
        if wp.get_device().is_cuda and self.solver_mpm.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_mpm()
            self.sand_graph = capture.graph

    def collect_collider_impulses(self):
        """从MPM求解器收集碰撞冲量"""
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.solver_mpm.collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self._n_colliders = int(n_colliders)
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    def simulate_leg(self):
        """物块仿真：执行物块动力学计算"""
        for substep in range(self.sim_substeps):

            self.state_0.clear_forces()  # 物块相关：清除上一步的外力

            # === [NEW] DMP CONTROL (2D) BEGIN =================================
            # 1) 根据“上一子步”的足端接触力更新 finish_flag（类似 env.py 的阈值逻辑）。
            #    设计意图：当足端已经明显触地（法向力大）或摩擦很大（切向力大）时，
            #    停止继续对对应维度施加 DMP 外力项，避免持续“压/拉”造成抖动。
            #
            #    这里用空间力的线性部分近似足端受力：
            #    - fz 作为法向（沿 z 轴），并做 max(0, fz) 只关心“向上顶”的力。
            #    - fx/fy 的合力作为切向（平面内）。
            fx, fy, fz = self._last_foot_force_world
            f_n = max(0.0, float(fz))
            f_t = float(np.sqrt(fx * fx + fy * fy))
            # finish_flag 的简单策略：
            # - 3D（推荐：HAA/HFE/KFE）：HAA 用切向阈值；HFE/KFE 用法向阈值。
            # - 其它维度：默认全用法向阈值，若维度>=2 则最后一维用切向阈值。
            self.finish_flag.fill(0)
            if self.dmp_n_dims >= 3:
                self.finish_flag[0] = 1 if f_t > self.dmp_ft_max else 0
                self.finish_flag[1] = 1 if f_n > self.dmp_fn_max else 0
                self.finish_flag[2] = 1 if f_n > self.dmp_fn_max else 0
                if self.dmp_n_dims > 3:
                    self.finish_flag[3:] = self.finish_flag[2]
            else:
                self.finish_flag[:] = 1 if f_n > self.dmp_fn_max else 0
                if self.dmp_n_dims >= 2:
                    self.finish_flag[-1] = 1 if f_t > self.dmp_ft_max else self.finish_flag[-1]

            if self.leg_controller == "dmp":
                # 2) env.py: force_eta = clip(action)*scale
                #    - action 限幅：保证输入在可控范围内。
                #    - 乘以 scale：把 [-1, 1] 的归一化输入映射到实际使用的耦合强度。
                action = np.clip(self.dmp_action, self.dmp_action_bound[0], self.dmp_action_bound[1])
                force_eta = action * self.dmp_force_scale

                # 3) 将本子步的 force_eta/finish_flag 写入 coupling_term。
                #    DMP.step() 内部会回调 coupling(y, yd) 得到 (cd, cdd)。
                self._dmp_coupling.force_eta = force_eta
                self._dmp_coupling.finish_flag = self.finish_flag

                # 4) DMP 前进一步，得到新的关节目标。
                self.dmp_pos, self.dmp_vel = self.dmp.step(
                    self.dmp_pos,
                    self.dmp_vel,
                    coupling_term=self._dmp_coupling,
                )
            else:
                # script 模式：手动分段轨迹（抬腿->下压->保持），并支持触地后停在当前/保持阶段。
                cycle_time = float(self.script_cycle_time)
                lift_ratio = float(np.clip(self.script_lift_ratio, 0.0, 1.0))
                press_ratio = float(np.clip(self.script_press_ratio, lift_ratio, 1.0))
                stop_on_contact = bool(self.script_stop_on_contact)
                beta = float(self.script_exp_beta)

                start = np.asarray(self._script_start, dtype=np.float64).reshape(self.dmp_n_dims)
                lift_goal = np.asarray(self._script_lift_goal, dtype=np.float64).reshape(self.dmp_n_dims)
                press_goal = np.asarray(self._script_press_goal, dtype=np.float64).reshape(self.dmp_n_dims)

                def _exp_ease(u: float) -> float:
                    u = float(np.clip(u, 0.0, 1.0))
                    if beta <= 0.0:
                        return u
                    denom = 1.0 - float(np.exp(-beta))
                    if denom <= 1e-12:
                        return u
                    return float((1.0 - np.exp(-beta * u)) / denom)

                # 触地后冻结：一旦触发，则直接进入保持阶段
                if stop_on_contact and ((f_n > self.dmp_fn_max) or (f_t > self.dmp_ft_max)):
                    phase = 1.0
                else:
                    t_now = float(self.sim_time + float(substep) * float(self.sim_dt))
                    t = float(t_now - float(getattr(self, "_script_t0", 0.0)))
                    if cycle_time <= 1e-6:
                        phase = 1.0
                    else:
                        phase = float(np.clip(t / cycle_time, 0.0, 1.0))

                if phase <= lift_ratio and lift_ratio > 1e-8:
                    u = _exp_ease(phase / lift_ratio)
                    q_tgt = start + (lift_goal - start) * u
                elif phase <= press_ratio and (press_ratio - lift_ratio) > 1e-8:
                    u = _exp_ease((phase - lift_ratio) / (press_ratio - lift_ratio))
                    q_tgt = lift_goal + (press_goal - lift_goal) * u
                else:
                    q_tgt = press_goal

                self.dmp_vel = (q_tgt - getattr(self, "_script_prev_target", q_tgt)) / float(self.sim_dt)
                self._script_prev_target = np.asarray(q_tgt, dtype=np.float64).copy()
                self.dmp_pos = np.asarray(q_tgt, dtype=np.float64).copy()

            # 5) 写入 Newton 控制目标（XPBD 会读取 joint_target_pos/joint_target_vel）。
            #    - 这里把 numpy(float64) 转成 float32，并通过 assign() 写入 device。
            #    - 关节目标的维度必须与 model.joint_dof_count 匹配；此例仅控制 hip/knee 两维。
            # 只更新 DMP 控制的 DOF，其余 DOF 保持初始目标（等价锁定）。
            target_pos = np.array(self._dmp_ref_joint_q, dtype=np.float64, copy=True)
            target_vel = np.zeros_like(target_pos)
            target_pos[self.dmp_dof_indices] = np.asarray(self.dmp_pos, dtype=np.float64).reshape(self.dmp_n_dims)
            target_vel[self.dmp_dof_indices] = np.asarray(self.dmp_vel, dtype=np.float64).reshape(self.dmp_n_dims)
            self.control.joint_target_pos.assign(target_pos.astype(np.float32).tolist())
            self.control.joint_target_vel.assign(target_vel.astype(np.float32).tolist())
            # === [NEW] DMP CONTROL (2D) END ===================================

            # 计算沙子施加在刚体上的力
            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            # 保存施加的力以便稍后减去
            self.body_sand_forces.assign(self.state_0.body_f)

            # === [NEW] DMP CONTROL (2D) BEGIN =================================
            # 记录足端线性力（世界坐标），用于“下一子步”的 finish_flag。
            # 说明：body_sand_forces 是每个刚体的空间力（6 维）：[fx, fy, fz, tx, ty, tz]。
            # 这里取 [:3] 只要线性力。
            self._last_foot_force_world = self.body_sand_forces.numpy()[self.foot][:3].astype(np.float32)
            # === [NEW] DMP CONTROL (2D) END ===================================

            self.viewer.apply_forces(self.state_0)  # 应用用户交互力

            # --- 直接在主模型上进行碰撞检测 ---
            self.contacts = self.model.collide(self.state_0)
            
            # 物块求解器步进
            self.solver_block.step(self.state_0, self.state_1, control=self.control, contacts=self.contacts, dt=self.sim_dt)

            # 交换双缓冲状态，保证下一次迭代读取到最新状态
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def simulate_mpm(self):
        """沙子仿真：执行 MPM 颗粒材料动力学计算"""
        
        # 从刚体速度中减去先前施加的冲量
        if self.state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.state_0.body_q.shape,
                inputs=[
                    self.frame_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state_0.body_q, # 结果直接写回 sand_state_0
                    self.sand_state_0.body_qd,
                ],
            )

        # 沙子相关：MPM 求解器主要步骤
        # 1. 粒子到网格传输（P2G）：将粒子质量/动量汇总到网格点
        # 2. 网格动力学求解：在网格上计算内力、外力，求解速度更新
        # 3. 网格到粒子传输（G2P）：将网格速度插值回粒子
        # 4. 粒子位置更新和材料状态更新
        self.solver_mpm.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)

        # 保存冲量以便施加回刚体
        self.collect_collider_impulses()

        # 统计：当前帧（MPM step 之后）足端与粒子交互力
        self.foot_particle_force.zero_()
        if self._n_colliders > 0:
            wp.launch(
                compute_single_body_contact_force,
                dim=self._n_colliders,
                inputs=[
                    self.frame_dt,
                    int(self.foot),
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_body_id,
                    self.foot_particle_force,
                ],
            )
            self.foot_particle_force_world = self.foot_particle_force.numpy()[0].astype(np.float32)

    def simulate(self):
        """主仿真步骤：每帧调用一次"""
        # ========== 执行仿真步骤 ==========
        # 1. 执行物块仿真
        self.simulate_leg()

        # 2. 执行沙子仿真
        self.simulate_mpm()

    
    def step(self):
        """设置CUDA图优化以加速仿真"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_leg()

        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_mpm()

        self.sim_time += self.frame_dt
        
        # ========== 计算并记录总作用力 ==========
        # 这里记录“LF 足端与 MPM 粒子”的交互力（由 collider 冲量统计得到）。
        force_magnitude = float(np.linalg.norm(self.foot_particle_force_world))
        self.force_history.append(force_magnitude)
        self.time_history.append(self.sim_time)

        # headless(null viewer) 下：一旦检测到持续的非零足-粒子交互力，就提前结束运行。
        # 目的：快速验证“确实发生足地交互”，避免长时间跑仿真。
        if self._headless_null_viewer:
            if force_magnitude > 1e-6:
                self._contact_seen_frames += 1
            else:
                self._contact_seen_frames = 0
            if self._contact_seen_frames >= 3:
                print("\nContact detected; stopping early.")
                raise SystemExit(0)

    def render(self):
        """渲染函数：显示沙子粒子和障碍物"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)  # 记录当前沙子状态用于可视化
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.mpm_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.solver_mpm.collect_collider_impulses(self.sand_state_0)
            impulse_scale = 100.0
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses * impulse_scale,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

        # ========== 终端打印与F-t曲线更新 ==========
        if self.force_history:
            current_force = self.force_history[-1]
            # 使用 \r 和 end="" 实现单行刷新
            fx, fy, fz = self.foot_particle_force_world
            print(
                f"\rLF Particle Force: {current_force:8.2f} N  (fx={fx:7.2f}, fy={fy:7.2f}, fz={fz:7.2f})",
                end="",
            )

            if self.enable_plot and self.line is not None and self.ax is not None and self.fig is not None:
                # 更新曲线数据
                self.line.set_xdata(self.time_history)
                self.line.set_ydata(self.force_history)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def render_ui(self, imgui):
        """渲染UI界面"""
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    @staticmethod
    def _quat_rotate(q_xyz_w: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector(s) v by quaternion q=(x,y,z,w)."""
        q = np.asarray(q_xyz_w, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        q_xyz = q[..., :3]
        w = q[..., 3:4]
        t = 2.0 * np.cross(q_xyz, v)
        return v + w * t + np.cross(q_xyz, t)

    @classmethod
    def _transform_points(cls, xform: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply warp-style transform [px,py,pz,qx,qy,qz,qw] to points."""
        x = np.asarray(xform, dtype=np.float64).reshape(7)
        p = x[:3]
        q = x[3:]
        return cls._quat_rotate(q, pts) + p

    def _shape_vertices_local(self, model: newton.Model, shape_id: int) -> np.ndarray:
        """Get triangle-mesh vertices for a shape in its local frame (before shape_transform)."""

        geo_type = int(model.shape_type.numpy()[shape_id])
        scale = np.array(model.shape_scale.numpy()[shape_id], dtype=np.float64)

        if geo_type == int(newton.GeoType.MESH) or geo_type == int(newton.GeoType.CONVEX_MESH):
            src_mesh = model.shape_source[shape_id]
            return np.asarray(src_mesh.vertices, dtype=np.float64) * float(np.max(scale))

        if geo_type == int(newton.GeoType.PLANE):
            width = scale[0] if scale[0] > 0.0 else 1000.0
            length = scale[1] if scale[1] > 0.0 else 1000.0
            v, _i = newton.utils.create_plane_mesh(width, length)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.SPHERE):
            radius = float(scale[0])
            v, _i = newton.utils.create_sphere_mesh(radius)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.CAPSULE):
            radius, half_height = float(scale[0]), float(scale[1])
            v, _i = newton.utils.create_capsule_mesh(radius, half_height, up_axis=2)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.CYLINDER):
            radius, half_height = float(scale[0]), float(scale[1])
            v, _i = newton.utils.create_cylinder_mesh(radius, half_height, up_axis=2)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.CONE):
            radius, half_height = float(scale[0]), float(scale[1])
            v, _i = newton.utils.create_cone_mesh(radius, half_height, up_axis=2)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.BOX):
            half_extents = (float(scale[0]), float(scale[1]), float(scale[2]))
            v, _i = newton.utils.create_box_mesh(extents=half_extents)
            return np.asarray(v, dtype=np.float64)

        if geo_type == int(newton.GeoType.ELLIPSOID):
            rx, ry, rz = float(scale[0]), float(scale[1]), float(scale[2])
            v, _i = newton.utils.create_ellipsoid_mesh(rx, ry, rz)
            return np.asarray(v, dtype=np.float64)

        raise NotImplementedError(f"Unsupported shape type for contact probe: {geo_type}")

    def _build_body_collision_points_body_frame(self, model: newton.Model, body_id: int) -> np.ndarray:
        """Build a point cloud (vertices) for all COLLIDE_PARTICLES shapes on a body, in body frame."""

        if body_id not in model.body_shapes:
            return np.zeros((0, 3), dtype=np.float64)

        shape_flags = model.shape_flags.numpy()
        shape_ids = [int(s) for s in model.body_shapes[body_id] if (shape_flags[int(s)] & int(newton.ShapeFlags.COLLIDE_PARTICLES))]
        if not shape_ids:
            return np.zeros((0, 3), dtype=np.float64)

        shape_xforms = model.shape_transform.numpy()
        pts_all: list[np.ndarray] = []
        for sid in shape_ids:
            v_local = self._shape_vertices_local(model, sid)
            v_body = self._transform_points(shape_xforms[sid], v_local)
            pts_all.append(v_body)

        return np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3), dtype=np.float64)

    def _foot_min_z_from_state(self, state: newton.State) -> float:
        """Compute world min-z of the particle-colliding foot geometry."""
        if self._foot_collision_points_body.shape[0] == 0:
            # fallback: use body origin
            return float(state.body_q.numpy()[self.foot][2])
        body_xform = state.body_q.numpy()[self.foot]
        pts_world = self._transform_points(body_xform, self._foot_collision_points_body)
        return float(np.min(pts_world[:, 2]))

    def _auto_base_z_for_contact(self, robot_urdf: str, base_z: float, foot_link: str, terrain_z_top: float) -> float:
        """基于 URDF 运动学与地形高度，自动修正固定基座高度。

        目标：让足端(foot_link 对应 body 的原点)初始高度落在地形表面上方一个小裕量，
        再配合 DMP 下探确保必然发生足-粒子接触。
        """

        tmp_builder = newton.ModelBuilder()
        tmp_builder.add_urdf(
            robot_urdf,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, float(base_z)), q=wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        tmp_model = tmp_builder.finalize()
        tmp_state = tmp_model.state()
        newton.eval_fk(tmp_model, tmp_state.joint_q, tmp_state.joint_qd, tmp_state)

        if not hasattr(tmp_builder, "body_key"):
            return base_z

        body_keys = [str(k) for k in tmp_builder.body_key]
        if foot_link not in body_keys:
            foot_found = next((k for k in body_keys if "FOOT" in k), None)
            if foot_found is None:
                foot_found = next((k for k in body_keys if "SHANK" in k), None)
            if foot_found is None and body_keys:
                foot_found = body_keys[-1]
            if foot_found is None:
                return base_z
            foot_link = foot_found

        foot_id = int(body_keys.index(foot_link))

        # 用“发生粒子碰撞的几何最低点”来校准高度，避免小腿/电机被算进足端高度
        tmp_points_body = self._build_body_collision_points_body_frame(tmp_model, foot_id)
        if tmp_points_body.shape[0] == 0:
            foot_min_z = float(tmp_state.body_q.numpy()[foot_id][2])
        else:
            body_xform = tmp_state.body_q.numpy()[foot_id]
            pts_world = self._transform_points(body_xform, tmp_points_body)
            foot_min_z = float(np.min(pts_world[:, 2]))

        # 初始足底离地有裕量：让“足底最低点”在地形表面上方
        desired_min_z = float(terrain_z_top + 0.08)
        dz = desired_min_z - foot_min_z
        base_z_new = float(base_z + dz)
        # 防止放得过低
        base_z_new = max(base_z_new, float(terrain_z_top + 0.10))
        return base_z_new

    # === [NEW] DMP CONTROL (2D) BEGIN =========================================
    def _init_dmp_joint_space(self, options, start_y: np.ndarray) -> DMP:
        """创建关节空间 DMP（维度 = 选定关节数）。

        Args:
            options: 命令行参数对象。
            start_y: 当前关节角（2维，rad）。

        Returns:
            DMP: 配置好的 DMP。
        """

        # 参考轨迹（借鉴 getInitialDMP.py 思路）：
        # - 用“指数型缓入”生成一个单调的下压轨迹（down-press），并在后段保持（hold）。
        # - 为了适配当前模型的关节轴/安装姿态，这里不硬编码 hip/knee 的正负方向，
        #   而是用有限差分看“哪个方向能让足端更接近地面(更低的 z)”，再自动选择 goal。
        #
        # 可调参数（均可通过 options 注入；本文件 argparse 未显式注册则走默认值）：
        # - dmp_execution_time: 轨迹总时长(s)
        # - dmp_dt: DMP 内部积分步长(s)
        # - dmp_n_steps: imitate 采样点数
        # - dmp_n_weights: 基函数权重数
        # - dmp_press_ratio: 下压阶段占比(0~1)，剩余阶段保持 goal
        # - dmp_press_hip / dmp_press_knee: 两个关节的“下压”幅度(rad)
        # - dmp_exp_beta: 指数缓入强度（越大越快接近 goal）
        exec_time = float(getattr(options, "dmp_execution_time", 1.5))
        # 默认让 DMP 的积分步长跟随子步 dt，避免时间尺度不匹配。
        dt = float(getattr(options, "dmp_dt", float(self.sim_dt)))
        n = int(getattr(options, "dmp_n_steps", 120))
        n_w = int(getattr(options, "dmp_n_weights", 20))
        press_ratio = float(getattr(options, "dmp_press_ratio", 0.9))
        press_ratio = float(np.clip(press_ratio, 0.05, 1.0))
        beta = float(getattr(options, "dmp_exp_beta", 2.0))

        n_dims = int(self.dmp_n_dims)
        start = np.asarray(start_y, dtype=np.float64).reshape(n_dims)

        # 每个关节的“下压”幅度（rad）。默认针对 LF 腿：HAA/HFE/KFE。
        # - HAA 通常幅度小一些（侧摆）
        # - HFE/KFE 用于抬/压腿
        default_press = {
            "LF_HAA": 0.25,
            "LF_HFE": 0.4,
            "LF_KFE": 0.8,
        }
        base_delta = np.zeros(n_dims, dtype=np.float64)
        for i, name in enumerate(self.dmp_joint_names):
            key = str(name)
            # 允许通过 options 覆盖：dmp_press_<jointname_lower>
            opt_key = f"dmp_press_{key.lower()}"
            if hasattr(options, opt_key):
                base_delta[i] = float(getattr(options, opt_key))
            else:
                base_delta[i] = float(default_press.get(key, 0.4))

        def _foot_min_z_from_joint(q: np.ndarray) -> float:
            tmp_state = self.model.state()
            # 在整机关节向量上仅替换 DMP 的两个 DOF，其他 DOF 保持初始值。
            q_full = np.array(self._dmp_ref_joint_q, dtype=np.float64, copy=True)
            q_full[self.dmp_dof_indices] = np.asarray(q, dtype=np.float64).reshape(n_dims)
            tmp_state.joint_q.assign(q_full.tolist())
            if tmp_state.joint_qd is not None:
                qd_len = int(tmp_state.joint_qd.shape[0])
                if qd_len > 0:
                    tmp_state.joint_qd.assign([0.0] * qd_len)
            newton.eval_fk(self.model, tmp_state.joint_q, tmp_state.joint_qd, tmp_state)
            return float(self._foot_min_z_from_state(tmp_state))

        # 用有限差分决定每个 DOF 的“下压方向”（使足端 z 降低的方向）
        eps = 1e-3
        z0 = _foot_min_z_from_joint(start)
        sign = np.ones(n_dims, dtype=np.float64)
        for j in range(n_dims):
            dq = np.zeros(n_dims, dtype=np.float64)
            dq[j] = eps
            z_plus = _foot_min_z_from_joint(start + dq)
            z_minus = _foot_min_z_from_joint(start - dq)
            if z_plus < z_minus:
                sign[j] = 1.0
            else:
                sign[j] = -1.0

        goal = start + sign * base_delta

        # 额外保险：把“足底最低点”下探到地形表面以下极小深度（确保接触但不把小腿埋入）。
        target_contact_z = float(self.terrain_z_top - 0.005)
        for scale in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
            cand = start + sign * base_delta * scale
            z_cand = _foot_min_z_from_joint(cand)
            if z_cand < min(z0, target_contact_z):
                goal = cand
                break

        def _exp_ease(u: np.ndarray) -> np.ndarray:
            # u in [0,1] -> s in [0,1]
            if beta <= 0.0:
                return u
            denom = 1.0 - np.exp(-beta)
            if denom <= 1e-12:
                return u
            return (1.0 - np.exp(-beta * u)) / denom

        T = np.linspace(0.0, exec_time, n, dtype=np.float64)
        phase = np.linspace(0.0, 1.0, n, dtype=np.float64)

        u = np.clip(phase / press_ratio, 0.0, 1.0)
        s = _exp_ease(u)
        # press_ratio 之后保持：s 已经被 clip 到 1
        Y = (start[None, :] + (goal - start)[None, :] * s[:, None]).astype(np.float64)

        dmp = DMP(
            n_dims=n_dims,
            execution_time=exec_time,
            dt=dt,
            n_weights_per_dim=n_w,
            smooth_scaling=True,
        )
        # 1) imitate：从 (T, Y) 学到一个可泛化的 DMP 表达。
        dmp.imitate(T, Y)
        # 2) configure：设置起点/终点。此处用参考轨迹的首末点。
        dmp.configure(start_y=Y[0], goal_y=Y[-1])
        return dmp
    # === [NEW] DMP CONTROL (2D) END ===========================================

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """根据噪声函数在指定区域生成三种不同类型的粒子：沙、雪、泥"""
        # 1. test 原始方法：生成所有粒子
        all_ids = Example._spawn_particles(builder, args,
                                           particle_lo=np.array([-0.5, -0.5, 0.0]),
                                           particle_hi=np.array([ 0.5,  0.5, 0.25]),
                                           density=1000,   # 先随便设一个基础密度，稍后会被替换
                                           flags=newton.ParticleFlags.ACTIVE)

        # 2. 使用噪声将粒子分类
        sand_ids, snow_ids, mud_ids, block_ids = Example._classify_by_noise(builder, all_ids)

        # 3. 为三种粒子分配不同密度（来自 mpm_block）
        Example._assign_density(builder, sand_ids, 2500)  # sand
        Example._assign_density(builder, snow_ids,  300)  # snow
        Example._assign_density(builder, mud_ids,  1000)  # mud
        Example._assign_density(builder, block_ids, 2700)  # block

        # 4. 返回粒子 ID
        return np.array(sand_ids), np.array(snow_ids), np.array(mud_ids), np.array(block_ids)

    #   分类粒子 → 沙、雪、泥（基于噪声并确保三类都存在）
    @staticmethod
    def _classify_by_noise(builder, ids):

        freq = 1.2
        octaves = 4

        noise_vals = []

        # 1. 计算噪声值
        for pid in ids:
            q = builder.particle_q[pid]
            n = noise.pnoise3(q[0]*freq, q[1]*freq, q[2]*freq, octaves=octaves)
            n = (n + 1) * 0.5
            noise_vals.append(n)

        noise_vals = np.array(noise_vals)

        # 2. 三等分阈值（避免某一类为 0）
        t1 = np.quantile(noise_vals, 0.25)
        t2 = np.quantile(noise_vals, 0.5)
        t3 = np.quantile(noise_vals, 0.75)

        sand = []
        snow = []
        mud = []
        block = []

        # 3. 分类
        for pid, n in zip(ids, noise_vals):
            if n <= t1:
                sand.append(pid)
            elif n <= t2:
                snow.append(pid)
            elif n <= t3:
                mud.append(pid)
            else:
                block.append(pid)

        return sand, snow, mud, block

    #   为一类粒子设置密度：根据体积反推出 mass
    @staticmethod
    def _assign_density(builder, particle_ids, density):

        for pid in particle_ids:
            radius = builder.particle_radius[pid]
            volume = (4.0/3.0) * np.pi * radius**3
            mass = density * volume
            builder.particle_mass[pid] = mass

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, args, particle_lo, particle_hi, density, flags):
        """沙子相关：根据命令行参数在指定区域生成沙子粒子"""
        # density = args.density          # 沙子密度
        voxel_size = float(getattr(args, "voxel_size", 0.03))   # MPM 网格体素大小

        particles_per_cell = 3.0  # 每个体素内的粒子数量
        # particle_lo = np.array(args.emit_lo)  # 沙子生成区域下界
        # particle_hi = np.array(args.emit_hi)  # 沙子生成区域上界
        
        # 沙子相关：根据体素大小计算粒子网格分辨率
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        # 计算单个粒子属性
        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(cell_volume * density) # 注意这里是 cell_volume, 不是 np.prod(cell_volume)

        begin_id = len(builder.particle_q)
        # 使用官方推荐的辅助函数
        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius, # 扰动强度
            radius_mean=radius,
            flags=flags,
        )

        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=int)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = newton.examples.create_parser()

    # ========== 场景配置参数 ==========
    # 交互相关：静态障碍物类型选择
    parser.add_argument("--collider", default="none", choices=["cube", "none"], type=str)
    # 沙子相关：粒子生成区域
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.25])
    # 沙子相关：重力设置
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    # 仿真控制：帧率和子步
    parser.add_argument("--fps", type=float, default=160)
    parser.add_argument("--substeps", type=int, default=32)

    # ========== 沙子材料属性参数 ==========
    parser.add_argument("--density", type=float, default=2500.0)              # 沙子密度
    parser.add_argument("--air-drag", type=float, default=1.0)                # 空气阻力
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0) # 临界状态参数

    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e15)  # 杨氏模量
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)     # 泊松比
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.48)   # 摩擦系数
    parser.add_argument("--damping", type=float, default=0.0)                  # 阻尼
    parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e5) # 屈服压力
    parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0) # 拉伸屈服比
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)      # 屈服应力
    parser.add_argument("--hardening", type=float, default=0.0)                # 硬化参数

    # ========== MPM 求解器参数 ==========
    parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
    parser.add_argument("--solver", "-s", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"])
    parser.add_argument("--transfer-scheme", "-ts", type=str, default="pic", choices=["apic", "pic"])
    parser.add_argument("--strain-basis", "-sb", type=str, default="P0", choices=["P0", "Q1"])

    parser.add_argument("--max-iterations", "-it", type=int, default=50)      # 最大迭代次数
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)     # 收敛容差
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)        # 网格体素大小

    # ========== [NEW] DMP CONTROL (2D) 参数 ==========
    # 说明：这部分参数主要控制“外力耦合项”和“停止阈值”。
    # - dmp-action：2 维输入（hip/knee），会在 simulate_leg() 中做 clip 后乘 dmp-force-scale。
    # - dmp-fn-max / dmp-ft-max：用足端受力估计法向/切向阈值，超过则置 finish_flag。
    # - joint-target-ke/kd：XPBD 关节目标的等效 PD 增益。
    parser.add_argument(
        "--dmp-action",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="DMP action (eta), length matches --dmp-joints (e.g. 3 for LF_HAA/LF_HFE/LF_KFE)",
    )
    parser.add_argument("--dmp-force-scale", type=float, default=500.0, help="Scale for force_eta = action*scale")
    parser.add_argument("--dmp-fn-max", type=float, default=400.0, help="Normal-force threshold for finish_flag")
    parser.add_argument("--dmp-ft-max", type=float, default=400.0, help="Tangential-force threshold for finish_flag")
    parser.add_argument("--joint-target-ke", type=float, default=2000.0, help="XPBD joint target stiffness")
    parser.add_argument("--joint-target-kd", type=float, default=50.0, help="XPBD joint target damping")

    # ========== 机器人 URDF 导入参数 ==========
    _default_urdf = os.path.join(os.path.dirname(__file__), "el_mini", "urdf", "el_mini.urdf")
    parser.add_argument("--robot-urdf", type=str, default=_default_urdf, help="URDF path for the robot")
    parser.add_argument("--robot-floating", action="store_true", help="Import robot with a floating base")
    parser.add_argument("--robot-base-z", type=float, default=0.3, help="Base height (z) for imported robot")
    parser.add_argument("--foot-link", type=str, default="LF_SHANK", help="Body(link) name used as the foot for force readout")
    parser.add_argument(
        "--dmp-joints",
        type=str,
        nargs="+",
        default=["LF_HAA", "LF_HFE", "LF_KFE"],
        help="1-DoF joint names controlled by the DMP (default: LF_HAA LF_HFE LF_KFE)",
    )

    # ========== 手动单腿动作脚本（可选） ==========
    # 用于不通过 RL、也不想改 DMP 参考轨迹时，直接手动指定“抬腿->下压->保持”的动作模式。
    parser.add_argument(
        "--leg-controller",
        type=str,
        default="dmp",
        choices=["dmp", "script"],
        help="Leg controller mode: 'dmp' (default) or 'script' (lift->press->hold)",
    )
    parser.add_argument("--script-cycle-time", type=float, default=1.2, help="Total duration (s) for script motion")
    parser.add_argument(
        "--script-lift-ratio",
        type=float,
        default=0.25,
        help="Lift phase ratio in [0,1] (start->lift_goal)",
    )
    parser.add_argument(
        "--script-press-ratio",
        type=float,
        default=0.65,
        help="Press phase end ratio in [lift_ratio,1] (lift_goal->press_goal)",
    )
    parser.add_argument(
        "--script-lift-scale",
        type=float,
        default=0.6,
        help="Lift amplitude scale (relative to per-joint dmp_press_*)",
    )
    parser.add_argument(
        "--script-press-scale",
        type=float,
        default=1.0,
        help="Press amplitude scale (relative to per-joint dmp_press_*)",
    )
    parser.add_argument(
        "--script-lift-delta",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-joint lift amplitude (rad), length matches --dmp-joints",
    )
    parser.add_argument(
        "--script-press-delta",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-joint press amplitude (rad), length matches --dmp-joints",
    )
    parser.add_argument(
        "--script-stop-on-contact",
        type=int,
        choices=[0, 1],
        default=1,
        help="Stop and hold once contact exceeds thresholds (1=yes, 0=no)",
    )
    parser.add_argument(
        "--script-exp-beta",
        type=float,
        default=2.0,
        help="Exponential easing beta (bigger -> faster approach to goal)",
    )

    # 解析参数并初始化查看器
    viewer, args = newton.examples.init(parser)

    # 创建示例并运行
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    except SystemExit:
        pass

    # headless 情况下，确保有一行可被日志捕获的输出
    if hasattr(example, "force_history") and example.force_history:
        print("\nMax LF particle force (N):", float(np.max(example.force_history)))