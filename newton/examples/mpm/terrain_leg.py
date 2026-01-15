from __future__ import annotations
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

        # ========== 沙子-物块交互系统构建 ==========
        builder = newton.ModelBuilder()

        # 交互相关：添加地面作为沙子的底部边界
        ground_shape = builder.add_ground_plane()

        # ========== 单腿系统构建 ==========
        # 连杆尺寸
        link1_hx = 0.1
        link1_hy = 0.02
        link1_hz = 0.02

        link2_hx = 0.15
        link2_hy = 0.015
        link2_hz = 0.015

        # 足端半径
        foot_radius = 0.02

        mass1 = 3.0
        mass2 = 3.0
        mass3 = 0.1
        
        # 创建大腿
        thigh = builder.add_link()
        builder.add_shape_box(
            thigh, hx=link1_hx, hy=link1_hy, hz=link1_hz, 
            cfg=newton.ModelBuilder.ShapeConfig(density=mass1/(8*link1_hx*link1_hy*link1_hz))
            )

        # 创建小腿
        shank = builder.add_link()
        builder.add_shape_box(
            shank, hx=link2_hx, hy=link2_hy, hz=link2_hz, 
            cfg=newton.ModelBuilder.ShapeConfig(density=mass2/(8*link2_hx*link2_hy*link2_hz))
            )
        # 创建足端
        self.foot = builder.add_link()
        builder.add_shape_sphere(
            self.foot, radius=foot_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=mass3/(4/3*wp.pi*foot_radius*foot_radius*foot_radius))
            )

        # 关节设置
        base_pos = wp.vec3(0.0, 0.0, 0.6)
        base_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        
        # 髋关节(Hip joint) - 连接世界和大腿
        hip_joint = builder.add_joint_revolute(
            parent=-1,  # -1 表示世界
            child=thigh,
            axis=wp.vec3(0.0, 1.0, 0.0), # 绕 Y 轴旋转
            parent_xform=wp.transform(p=base_pos, q=base_rot),
            child_xform=wp.transform(p=wp.vec3(-link1_hx, 0.0, 0.0), q=wp.quat_identity()),
        )

         # 膝关节 (Knee joint) - 连接大腿和小腿
        knee_joint = builder.add_joint_revolute(
            parent=thigh,
            child=shank,
            axis=wp.vec3(0.0, 1.0, 0.0), # 绕 Y 轴旋转
            parent_xform=wp.transform(p=wp.vec3(link1_hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-link2_hx, 0.0, 0.0), q=wp.quat_identity()),
        )
        
         # 踝关节 (Ankle joint) - 连接小腿和足端
        ankle_joint = builder.add_joint_fixed(
            parent=shank,
            child=self.foot,    
            parent_xform=wp.transform(p=wp.vec3(link2_hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        # 将该腿的关节打包成一个 articulation（关节索引必须是连续且递增的）
        builder.add_articulation([hip_joint, knee_joint, ankle_joint], key="leg")
        
        # ========== 沙子模型构建 ==========
        mpm_builder = newton.ModelBuilder()
        # 沙子相关：在指定区域生成 MPM 粒子
        sand_particles, snow_particles, mud_particles, block_particles = Example.emit_particles(mpm_builder, options)

        # ========== 物理模型最终化 ==========
        self.model = builder.finalize()
        self.mpm_model = mpm_builder.finalize()

        # 沙子相关：设置沙子的摩擦属性
        self.mpm_model.particle_mu = options.friction_coeff
        self.mpm_model.particle_kd = 0.0
        self.mpm_model.particle_ke = 1.0e15

        sand_particles = wp.array(sand_particles, dtype=int, device=self.mpm_model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=self.mpm_model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=self.mpm_model.device)
        block_particles = wp.array(block_particles, dtype=int, device=self.mpm_model.device)

        # 沙子相关：设置重力（影响沙子下落和堆积）
        self.model.set_gravity(options.gravity)
        self.mpm_model.set_gravity(options.gravity)

        # ========== 沙子 MPM 求解器配置 ==========
        # 沙子相关：创建 MPM 求解器选项并复制命令行参数
        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

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
        # dmp_action：2 维输入，分别对应 hip/knee 的外力项（实际会乘上 dmp_force_scale）。
        self.dmp_action = np.array(getattr(options, "dmp_action", [0.0, 0.0]), dtype=np.float32)
        # finish_flag：2 维停止标志，后续在 simulate_leg() 里根据足端力更新。
        self.finish_flag = np.zeros(2, dtype=np.int32)
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
        initial_q = self.state_0.joint_q.numpy().astype(np.float32)
        self.dmp = self._init_dmp_joint_space(options, initial_q)
        # DMP 内部使用 float64 计算更稳（movement_primitives 也常用 float64）。
        self.dmp_pos = self.dmp.start_y.astype(np.float64)
        self.dmp_vel = np.zeros_like(self.dmp_pos, dtype=np.float64)

        # 耦合项对象：用于把 action/停止信号注入到 DMP 的 step() 中。
        self._dmp_coupling = self._DMPCouplingTerm(n_dims=2)

        # 初始把 XPBD 目标设置为当前关节角/速度。
        # 目的：避免第一帧就从默认目标（可能为 0）瞬间拉到当前姿态，引入不必要的冲击。
        self.control.joint_target_pos.assign(self.dmp_pos.tolist())
        self.control.joint_target_vel.assign(self.dmp_vel.tolist())

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
        self.collect_collider_impulses()

        # 从碰撞体索引到刚体索引的映射
        self.collider_body_id = mpm_model.collider.collider_body_index

        # 沙子施加到刚体上的每个刚体的力和扭矩
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

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
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.time_history, self.force_history)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.set_title("Interaction Force vs. Time")
        self.ax.grid(True)
        self.capture()

    def capture(self):
        """设置CUDA图优化以加速仿真"""
        self.graph = None
        # [NEW] DMP CONTROL (2D): DMP 计算与 host<->device 数据交换不适配 CUDA Graph capture
        if hasattr(self, "dmp"):
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
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    def simulate_leg(self):
        """物块仿真：执行物块动力学计算"""
        for _ in range(self.sim_substeps):

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
            self.finish_flag[0] = 1 if f_n > self.dmp_fn_max else 0
            self.finish_flag[1] = 1 if f_t > self.dmp_ft_max else 0

            # 2) env.py: force_eta = clip(action)*scale
            #    - action 限幅：保证输入在可控范围内。
            #    - 乘以 scale：把 [-1, 1] 的归一化输入映射到实际使用的耦合强度。
            action = np.clip(self.dmp_action, self.dmp_action_bound[0], self.dmp_action_bound[1])
            force_eta = action * self.dmp_force_scale

            # 3) 将本子步的 force_eta/finish_flag 写入 coupling_term。
            #    DMP.step() 内部会回调 coupling(y, yd) 得到 (cd, cdd)。
            self._dmp_coupling.force_eta = force_eta
            self._dmp_coupling.finish_flag = self.finish_flag

            # 4) DMP 前进一步，得到新的关节目标（2维：hip/knee）。
            #    注意：这里是在“每个子步”都更新一次 DMP。
            #    这意味着 DMP 的 dt/执行时间等参数（见 _init_dmp_joint_space）
            #    应当与子步频率一致，否则会出现时间尺度不匹配。
            self.dmp_pos, self.dmp_vel = self.dmp.step(
                self.dmp_pos,
                self.dmp_vel,
                coupling_term=self._dmp_coupling,
            )

            # 5) 写入 Newton 控制目标（XPBD 会读取 joint_target_pos/joint_target_vel）。
            #    - 这里把 numpy(float64) 转成 float32，并通过 assign() 写入 device。
            #    - 关节目标的维度必须与 model.joint_dof_count 匹配；此例仅控制 hip/knee 两维。
            self.control.joint_target_pos.assign(self.dmp_pos.astype(np.float32).tolist())
            self.control.joint_target_vel.assign(self.dmp_vel.astype(np.float32).tolist())
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
        # body_sand_forces 包含线性和角力，我们只关心线性力（前3个分量）
        #           |力和力矩              |复制   |索引             |前三个
        force_vec = self.body_sand_forces.numpy()[self.foot][:3]
        # 这个三维向量的欧几里得范数（也就是它的模长或大小）
        force_magnitude = float(np.linalg.norm(force_vec))
        self.force_history.append(force_magnitude)
        self.time_history.append(self.sim_time)

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
            print(f"\rInteraction Force: {current_force:8.2f} N", end="")

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

    # === [NEW] DMP CONTROL (2D) BEGIN =========================================
    def _init_dmp_joint_space(self, options, start_y: np.ndarray) -> DMP:
        """创建二维 DMP（关节空间：hip/knee）。

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
        exec_time = float(getattr(options, "dmp_execution_time", 1.0))
        dt = float(getattr(options, "dmp_dt", 0.01))
        n = int(getattr(options, "dmp_n_steps", 120))
        n_w = int(getattr(options, "dmp_n_weights", 20))
        press_ratio = float(getattr(options, "dmp_press_ratio", 0.8))
        press_ratio = float(np.clip(press_ratio, 0.05, 1.0))
        beta = float(getattr(options, "dmp_exp_beta", 4.3))

        hip0, knee0 = float(start_y[0]), float(start_y[1])
        start = np.array([hip0, knee0], dtype=np.float64)

        hip_delta = float(getattr(options, "dmp_press_hip", 0.4))
        knee_delta = float(getattr(options, "dmp_press_knee", 0.8))
        base_delta = np.array([hip_delta, knee_delta], dtype=np.float64)

        def _foot_z_from_joint(q: np.ndarray) -> float:
            tmp_state = self.model.state()
            tmp_state.joint_q.assign([float(q[0]), float(q[1])])
            if tmp_state.joint_qd is not None:
                qd_len = int(tmp_state.joint_qd.shape[0])
                if qd_len > 0:
                    tmp_state.joint_qd.assign([0.0] * qd_len)
            newton.eval_fk(self.model, tmp_state.joint_q, tmp_state.joint_qd, tmp_state)
            return float(tmp_state.body_q.numpy()[self.foot][2])

        # 用有限差分决定每个 DOF 的“下压方向”（使足端 z 降低的方向）
        eps = 1e-3
        z0 = _foot_z_from_joint(start)
        sign = np.ones(2, dtype=np.float64)
        for j in range(2):
            dq = np.zeros(2, dtype=np.float64)
            dq[j] = eps
            z_plus = _foot_z_from_joint(start + dq)
            z_minus = _foot_z_from_joint(start - dq)
            if z_plus < z_minus:
                sign[j] = 1.0
            else:
                sign[j] = -1.0

        goal = start + sign * base_delta

        # 额外保险：若 goal 没有让足端更低，则逐步放大幅度再试几次
        for scale in (1.0, 1.5, 2.0):
            cand = start + sign * base_delta * scale
            if _foot_z_from_joint(cand) < z0:
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
            n_dims=2,
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
        voxel_size = args.voxel_size   # MPM 网格体素大小

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
    parser.add_argument("--dmp-action", type=float, nargs=2, default=[0.0, 0.0], help="2D DMP action (eta)")
    parser.add_argument("--dmp-force-scale", type=float, default=500.0, help="Scale for force_eta = action*scale")
    parser.add_argument("--dmp-fn-max", type=float, default=400.0, help="Normal-force threshold for finish_flag")
    parser.add_argument("--dmp-ft-max", type=float, default=400.0, help="Tangential-force threshold for finish_flag")
    parser.add_argument("--joint-target-ke", type=float, default=2000.0, help="XPBD joint target stiffness")
    parser.add_argument("--joint-target-kd", type=float, default=50.0, help="XPBD joint target damping")

    # 解析参数并初始化查看器
    viewer, args = newton.examples.init(parser)

    # 创建示例并运行
    example = Example(viewer, args)
    newton.examples.run(example, args)