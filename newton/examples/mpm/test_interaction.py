# uv run -m newton.examples.mpm.test_interaction
from __future__ import annotations

# pyright: reportInvalidTypeForm=false

import os
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from collections import deque

import noise
import numpy as np
import warp as wp

import math

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM

# --------------------------
# 状态机全局参数初始化
# --------------------------
# 状态定义
STATE_STEP = 1       # 状态1：沿三角步态轨迹迈步到候选点
STATE_CHECK = 2      # 状态2：多帧稳定性检测（Fz/滑动/沉陷）
STATE_LIFT = 3       # 状态3：回退前抬脚
STATE_BACKTRACK = 4  # 状态4：轨迹回退到下一个候选点
STATE_LOCK = 5       # 状态5：找到可靠点，锁定接触


def _try_import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    cid = collider_ids[i]

    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt
        x_wb = body_q[body_index]
        x_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(x_wb, x_com)

        wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    body_id = wp.tid()
    f = body_f[body_id]
    delta_v = dt * body_inv_mass[body_id] * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])
    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel
def compute_single_body_contact_force(
    dt: float,
    target_body: int,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    out_force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    cid = collider_ids[i]
    if cid < 0 or cid >= body_ids.shape[0]:
        return

    body_index = body_ids[cid]
    if body_index != target_body:
        return

    f_world = collider_impulses[i] / dt
    wp.atomic_add(out_force, 0, f_world)

@dataclass
class _LegSpec:
    name: str  # 例如："RF"
    hip_body: int  # 髋关节对应的 body id（例如 RF_HIP）
    foot_body: int  # 足端对应的 body id（优先 *_FOOT；不存在则回退 *_SHANK）
    foot_offset_local: np.ndarray  # shape=(3,), in foot_body local frame; when FOOT collapsed, this is SHANK->FOOT origin
    # side_sign: float  # 左腿=+1（+y），右腿=-1（-y），用于把横向偏移做成左右对称
    # phase_offset: float  # 相位偏移（tripod 两组相差 0.5）
    target_pos: wp.array  # shape=(1,), dtype=wp.vec3：给 IK 目标使用的缓存
    pos_obj: newton.ik.IKPositionObjective  # 足端位置目标
    foot_rel_base0: np.ndarray | None = None  # 足端在 BASE 局部坐标下的初始落点向量

class Example:
    def __init__(self, viewer, options):
        self.fps = float(getattr(options, "fps", 60))
        self.frame_dt = 1.0/float(self.fps)
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(options, "substeps", 4))
        self.sim_dt = float(self.frame_dt/self.sim_substeps)

        self.viewer = viewer
        self._headless_null_viewer = self.viewer.__class__.__name__.lower().endswith("null")
        self._contact_seen_frames = 0

        emit_hi = getattr(options, "emit_hi", [0.5, 0.5, 0.25])
        voxel_size = float(getattr(options, "voxel_size", 0.03))
        self.terrain_z_top = float(emit_hi[2]) + 0.5 * voxel_size

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,# 关节软边距（到限位前6cm开始缓冲）
            limit_ke=1.0e3,# 关节限位刚度（能否轻易突破限位）
            limit_kd=1.0e1,# 关节限位阻尼（到达限位时阻尼）
        )

        builder.default_joint_cfg.target_ke = float(getattr(options, "joint_target_ke", 2000.0))
        builder.default_joint_cfg.target_kd = float(getattr(options, "joint_target_kd", 50.0))

        builder.default_shape_cfg.ke = 5.0e4    # 碰撞弹性（刚度）
        builder.default_shape_cfg.kd = 5.0e2    # 碰撞阻尼
        builder.default_shape_cfg.kf = 1.0e3    # 摩擦刚度
        friction_coeff = float(getattr(options, "friction_coeff", 0.48))
        builder.default_shape_cfg.mu = friction_coeff

        # ground plane 仅作为 MPM 粒子边界：避免机器人同时踩在“刚体地面”上
        ground_shape = builder.add_ground_plane()
        if hasattr(builder, "shape_flags"):
            builder.shape_flags[ground_shape] = builder.shape_flags[ground_shape] & ~newton.ShapeFlags.COLLIDE_SHAPES

        nowpath = os.path.dirname(__file__)
        default_urdf = os.path.join(nowpath, "el_mini", "urdf", "el_mini.urdf")
        robot_urdf = str(getattr(options, "robot_urdf", default_urdf))
        self._robot_urdf_path = robot_urdf

        robot_floating = bool(getattr(options, "robot_floating", True))
        base_z = float(getattr(options, "robot_base_z", 1.0))

        foot_link = str(getattr(options, "foot_link", "LF_SHANK"))

        builder.add_urdf(
            robot_urdf,
            xform=wp.transform(wp.vec3(-0.18, -0.08 ,float(base_z)), wp.quat_identity()),
            floating=robot_floating,
            enable_self_collisions=False,# 开启自碰撞
            collapse_fixed_joints=True,# 合并固定关节
            ignore_inertial_definitions=False,# 忽略惯性参数
        )

        self._builder_joint_key = [str(k) for k in list(getattr(builder, "joint_key", []))]

        # 仅允许“足端” shapes 与粒子发生接触（其余 body 关闭 COLLIDE_PARTICLES），结构对齐 robot_terrain。
        if hasattr(builder, "shape_flags") and hasattr(builder, "body_key") and hasattr(builder, "body_shapes"):
            body_names = [str(x) for x in list(builder.body_key)]
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

            if not foot_bodies:
                for i, name in enumerate(body_names):
                    up = name.upper()
                    if ("FOOT" in up) or ("TOE" in up) or ("SOLE" in up):
                        foot_bodies.append(int(i))

            if (not foot_bodies) and (foot_link in builder.body_key):
                foot_bodies = [int(builder.body_key.index(foot_link))]

            foot_bodies = sorted(set(int(b) for b in foot_bodies))

            keep_shape_ids: list[int] = []
            if hasattr(builder, "shape_key"):
                for body in foot_bodies:
                    for sid in builder.body_shapes[body]:
                        key = str(builder.shape_key[sid]).upper()
                        if ("FOOT" in key) or ("TOE" in key) or ("SOLE" in key):
                            keep_shape_ids.append(int(sid))

            if not keep_shape_ids:
                for body in foot_bodies:
                    keep_shape_ids.extend(int(s) for s in builder.body_shapes[body])

            keep_shape_set = set(int(s) for s in keep_shape_ids)
            keep_shape_set.add(int(ground_shape))

            for body in range(builder.body_count):
                for shape in builder.body_shapes[body]:
                    if int(shape) not in keep_shape_set:
                        builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        # 记录足端/基座 body 索引（用于取力与参考坐标）
        self.foot = 0
        if hasattr(builder, "body_key") and foot_link in builder.body_key:
            self.foot = int(builder.body_key.index(foot_link))

        self.base = 0
        if hasattr(builder, "body_key"):
            if "BASE" in builder.body_key:
                self.base = int(builder.body_key.index("BASE"))
            else:
                keys = [str(k).upper() for k in list(builder.body_key)]
                for token in ("BASE", "TRUNK", "TORSO", "ROOT"):
                    idx = next((i for i, k in enumerate(keys) if token in k), None)
                    if idx is not None:
                        self.base = int(idx)
                        break

        # ------------------------------------------------------------------
        # 粒子地形（MPM）：参考 robot_terrain / move_on_terrain
        # ------------------------------------------------------------------
        mpm_builder = newton.ModelBuilder()
        sand_particles, snow_particles, mud_particles, block_particles = Example.emit_particles(mpm_builder, options)

        self.model = builder.finalize()
        self.mpm_model = mpm_builder.finalize()

        gravity = getattr(options, "gravity", [0.0, 0.0, -9.81])
        self.model.set_gravity(gravity)
        self.mpm_model.set_gravity(gravity)
        self.mpm_model.particle_mu = friction_coeff
        self.mpm_model.particle_kd = 0.0
        self.mpm_model.particle_ke = 1.0e15

        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # CPU 设备上，sparse grid 的 allocate_by_tiles 路径仅支持 CUDA。
        if hasattr(mpm_options, "grid_type") and (not wp.get_device().is_cuda):
            if getattr(mpm_options, "grid_type", None) == "sparse":
                mpm_options.grid_type = "fixed"

        mpm_model = SolverImplicitMPM.Model(self.mpm_model, mpm_options)

        sand_particles_wp = wp.array(sand_particles, dtype=int, device=self.mpm_model.device)
        snow_particles_wp = wp.array(snow_particles, dtype=int, device=self.mpm_model.device)
        mud_particles_wp = wp.array(mud_particles, dtype=int, device=self.mpm_model.device)
        block_particles_wp = wp.array(block_particles, dtype=int, device=self.mpm_model.device)

        # --- 雪的属性 (类雪材料) ---
        mpm_model.material_parameters.yield_pressure[snow_particles_wp].fill_(2.0e4)
        mpm_model.material_parameters.yield_stress[snow_particles_wp].fill_(1.0e3)
        mpm_model.material_parameters.tensile_yield_ratio[snow_particles_wp].fill_(0.05)
        mpm_model.material_parameters.friction[snow_particles_wp].fill_(0.1)
        mpm_model.material_parameters.hardening[snow_particles_wp].fill_(10.0)

        # --- 泥浆的属性 (类流体材料) ---
        mpm_model.material_parameters.yield_pressure[mud_particles_wp].fill_(1.0e10)
        mpm_model.material_parameters.yield_stress[mud_particles_wp].fill_(3.0e2)
        mpm_model.material_parameters.tensile_yield_ratio[mud_particles_wp].fill_(1.0)
        mpm_model.material_parameters.hardening[mud_particles_wp].fill_(2.0)
        mpm_model.material_parameters.friction[mud_particles_wp].fill_(0.0)

        # --- 石块的属性 (固体材料) ---
        mpm_model.material_parameters.yield_pressure[block_particles_wp].fill_(1.0e8)
        mpm_model.material_parameters.yield_stress[block_particles_wp].fill_(1.0e7)
        mpm_model.material_parameters.tensile_yield_ratio[block_particles_wp].fill_(0.8)
        mpm_model.material_parameters.friction[block_particles_wp].fill_(0.6)
        mpm_model.material_parameters.hardening[block_particles_wp].fill_(20.0)

        mpm_model.notify_particle_material_changed()
        mpm_model.setup_collider(model=self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control =  self.model.control()

        if self.model.joint_target_ke is not None and self.model.joint_target_kd is not None:
            ke = float(getattr(options, "joint_target_ke", 150.0))
            kd = float(getattr(options, "joint_target_kd", 5))
            self._set_ke_kd_for_nonfree_joints(self.model, ke=ke, kd=kd)

        self._default_target_pos_dof = self._build_default_target_pos_dof(self.model, self.state_0)
        dof_count = int(self.model.joint_dof_count)

        self._control_dtype_pos = self.control.joint_target_pos.dtype
        self._control_dtype_vel = self.control.joint_target_vel.dtype
        self._set_control_targets(
            target_pos_dof=self._default_target_pos_dof,
            target_vel_dof=np.zeros(dof_count,dtype=np.float64),
        )

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        auto_drop = bool(getattr(options, "auto_drop", True))
        if robot_floating and auto_drop:
            clearance = float(getattr(options, "drop_clearance", 0.002))
            self._drop_to_ground(self.model, self.state_0, ground_z=float(self.terrain_z_top), clearance=clearance)
            self.state_1.joint_q.assign(self.state_0.joint_q.numpy().tolist())
            if self.state_0.joint_qd is not None and self.state_1.joint_qd is not None:
                self.state_1.joint_qd.assign(self.state_0.joint_qd.numpy().tolist())
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # 刚体求解器：支持 MuJoCo 或 XPBD（由 CLI 选择）。
        self.rigid_solver = str(getattr(options, "rigid_solver", "mujoco")).strip().lower()

        # XPBD 下建议使用更“软”的关节目标增益（否则容易抖动）。
        if self.rigid_solver == "xpbd" and self.model.joint_target_ke is not None and self.model.joint_target_kd is not None:
            xpbd_ke = float(getattr(options, "xpbd_joint_target_ke", 600.0))
            xpbd_kd = float(getattr(options, "xpbd_joint_target_kd", 80.0))
            if xpbd_ke > 0.0 and xpbd_kd >= 0.0:
                self._set_ke_kd_for_nonfree_joints(self.model, ke=xpbd_ke, kd=xpbd_kd)
        self.solver_mujoco = newton.solvers.SolverMuJoCo(
            self.model,
            ls_parallel=True,
            njmax=50,
        )

        # XPBD 更稳的默认参数（可通过 CLI 覆盖）。
        xpbd_iterations = int(getattr(options, "xpbd_iterations", 80))
        xpbd_rigid_contact_relaxation = float(getattr(options, "xpbd_rigid_contact_relaxation", 0.3))
        xpbd_joint_linear_relaxation = float(getattr(options, "xpbd_joint_linear_relaxation", 0.3))
        xpbd_joint_angular_relaxation = float(getattr(options, "xpbd_joint_angular_relaxation", 0.3))
        xpbd_angular_damping = float(getattr(options, "xpbd_angular_damping", 0.4))

        self.solver_xpbd = newton.solvers.SolverXPBD(
            self.model,
            iterations=xpbd_iterations,
            rigid_contact_relaxation=xpbd_rigid_contact_relaxation,
            joint_linear_relaxation=xpbd_joint_linear_relaxation,
            joint_angular_relaxation=xpbd_joint_angular_relaxation,
            angular_damping=xpbd_angular_damping,
        )

        # MPM 状态与求解器
        self.sand_state_0 = self.mpm_model.state()
        self.solver_mpm = SolverImplicitMPM(mpm_model, mpm_options)
        self.solver_mpm.enrich_state(self.sand_state_0)

        # MPM 稳定性/排错开关：用于定位“粒子突然消失（NaN/Inf/飞出边界/全部 inactive）”的根因。
        self.mpm_substeps = int(max(1, getattr(options, "mpm_substeps", 1)))
        self._debug_mpm_health = bool(getattr(options, "debug_mpm_health", False))
        self._debug_mpm_health_every = int(max(1, getattr(options, "debug_mpm_health_every", 30)))
        self._debug_mpm_health_sample = int(max(0, getattr(options, "debug_mpm_health_sample", 0)))
        self._debug_mpm_health_stop = bool(getattr(options, "debug_mpm_health_stop", False))
        self._frame_index = 0

        # MPM -> 刚体耦合：碰撞冲量缓冲区
        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self._n_colliders = 0
        self.collect_collider_impulses()
        self.collider_body_id = mpm_model.collider.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)
        self.viewer.set_model(self.model)

        # 粒子渲染
        self.viewer.show_particles = True
        self.show_impulses = bool(getattr(options, "show_impulses", False))
        self.particle_render_colors = wp.full(
            self.mpm_model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.mpm_model.device
        )
        self.particle_render_colors[sand_particles_wp].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_render_colors[snow_particles_wp].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_render_colors[mud_particles_wp].fill_(wp.vec3(0.4, 0.25, 0.25))
        self.particle_render_colors[block_particles_wp].fill_(wp.vec3(0.5, 0.5, 0.5))

        # ------------------------------------------------------------------
        # 可视化：LF/RF 足端轨迹（可选）
        # ------------------------------------------------------------------
        self.show_foot_traj = bool(getattr(options, "show_foot_traj", False))
        self._foot_traj_maxlen = int(max(2,round(self.fps * 5.0)))
        self._traj_lf = deque(maxlen=self._foot_traj_maxlen)
        self._traj_rf = deque(maxlen=self._foot_traj_maxlen)

        # ------------------------------------------------------------------
        # 启动阶段：先原地站稳，再开始前进
        # ------------------------------------------------------------------
        self.settle_time = float(getattr(options, "settle_time", 0.8))

        # ------------------------------------------------------------------
        # 遥操作：控制 LF 三个关节的目标位置（每帧按键增量更新）
        # 6 个按键分别控制 3 个关节的正/反方向，且可同时按下实现耦合：
        #   i/k: 关节 0  + / -
        #   j/l: 关节 1  + / -
        #   u/o: 关节 2  + / -
        # 无输入时保持上一帧的目标不变（保证关节保持刚度，不“软下来”）。
        # ------------------------------------------------------------------
        self._target_pos_dof_cache = np.asarray(self._default_target_pos_dof, dtype=np.float64).copy()
        self._teleop_joint_speed = float(getattr(options, "teleop_joint_speed", 1.0))  # rad/s
        self._lf_teleop_joints = self._select_lf_teleop_joints(self.model)

        # Bezier 控制点的横向偏移（BASE 局部 y 方向；左腿自动取 +，右腿自动取 -）
        # 默认沿用之前的写法：p1/p2 轻微向外侧偏 0.1，避免内八/互相干涉。
        self.swing_y0 = float(getattr(options, "swing_y0", 0.0))
        self.swing_y1 = float(getattr(options, "swing_y1", 0.1))
        self.swing_y2 = float(getattr(options, "swing_y2", 0.1))
        self.swing_y3 = float(getattr(options, "swing_y3", 0.0))

        device = self.model.device

        # 用“浮动基座 free joint”的局部坐标系作为统一规划坐标系。
        # 注意：该 URDF 在 finalize 后 `model.body_key` 里不一定保留 "BASE" 这个 body。
        self._free_joint_q0 = self._find_free_joint_q0(self.model)
        
        leg_order = ["LF"]
        self.legs: list[_LegSpec] = []
        pos_objs: list[newton.ik.IKPositionObjective] = []

        for leg in leg_order:
            hip_name = f"{leg}_HIP"
            foot_name = f"{leg}_FOOT"
            hip_body = self._find_body_index(self.model, hip_name)
            foot_body = self._find_body_index_optional(self.model, foot_name)

            # 当 collapse_fixed_joints=True 时，*_FOOT 往往会被折叠进 *_SHANK。
            # 此时我们用 SHANK body + URDF 中 SHANK->FOOT fixed joint 的 origin 偏移来表示“足端点”。
            foot_offset_local = np.zeros(3, dtype=np.float64)
            if foot_body is None:
                foot_body = self._find_body_index(self.model, f"{leg}_SHANK")
                foot_offset_local = self._urdf_joint_origin_xyz(
                    self._robot_urdf_path,
                    parent_link=f"{leg}_SHANK",
                    child_link=f"{leg}_FOOT",
                )

            target_pos = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
            pos_obj = newton.ik.IKPositionObjective(
                link_index=int(foot_body),
                link_offset=wp.vec3(
                    float(foot_offset_local[0]),
                    float(foot_offset_local[1]),
                    float(foot_offset_local[2]),
                ),
                target_positions=target_pos,
                weight=10.0,
            )
            self.legs.append(
                _LegSpec(
                    name=leg,
                    hip_body=int(hip_body),
                    foot_body=int(foot_body),
                    foot_offset_local=foot_offset_local,
                    target_pos=target_pos,
                    pos_obj=pos_obj,
                )
            )
            pos_objs.append(pos_obj)

        # 统一“状态机足端”与“接触力统计足端”的 body id。
        # 之前 `self.foot` 来自 `--foot-link`（builder 阶段），而状态机读位姿用的是 `leg.foot_body`（model 阶段）。
        # 两者不一致时会导致：脚在动，但统计到的 Fz≈0，状态机一直卡在 FALL 下踩。
        lf_leg = next((l for l in self.legs if l.name == "LF"), None)
        self._foot_contact_body = int(lf_leg.foot_body) if lf_leg is not None else int(self.foot)
        self.foot = int(self._foot_contact_body)

        
        self._ik_joint_limits = newton.ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=1000.0,
        )
        self._ik_solver = newton.ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[*pos_objs, self._ik_joint_limits],
            optimizer=newton.ik.IKOptimizer.LM,
            jacobian_mode=newton.ik.IKJacobianMode.MIXED,
        )
        self._ik_solution_q = wp.zeros((1, int(self.model.joint_coord_count)), dtype=wp.float32, device=device)
        self._ik_iters = int(getattr(options, "ik_iters", 50))

        # 兜底：当前模型链上 Newton IK 可能返回不变解，导致状态机无法移动足端。
        # 这里准备一个纯数值 IK（有限差分雅可比 + 阻尼最小二乘）所需的 kinematic state。
        self._ik_numeric_state = self.model.state()
        if getattr(self._ik_numeric_state, "joint_qd", None) is not None:
            self._ik_numeric_state.joint_qd.zero_()
        self._ik_numeric_eps = float(getattr(options, "ik_numeric_eps", 1.0e-4))
        self._ik_numeric_damping = float(getattr(options, "ik_numeric_damping", 1.0e-4))
        self._ik_numeric_max_iters = int(getattr(options, "ik_numeric_max_iters", 4))

        # host 侧遥操作会导致 host<->device 数据交换，不适配 CUDA Graph capture
        self._disable_cuda_graph = True

        # ------------------------------------------------------------------
        # 足端-粒子接触力统计（实时读取 + 曲线）
        # ------------------------------------------------------------------
        self.foot_particle_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.foot_particle_force_world = np.zeros(3, dtype=np.float32)
        self._fx_filtered = 0.0
        self._fy_filtered = 0.0
        self._fz_filtered = 0.0
        self._f_filter_alpha = float(getattr(options, "fz_filter_alpha", 0.2))
        self._f_max = float(getattr(options, "force_max", 50.0))
        self.Fz_th = float(getattr(options, "fz_contact_threshold", 100.0))
        self._sm_last_fz: float | None = None
        self.Fz_ref_slide = float(getattr(options, "fz_ref_slide", 10.0))
        self.Kp_z = float(getattr(options, "kp_z", 800.0))
        # 状态机探索速度倍率（统一缩放 fall/slide/press/retract 的等效速度）
        self.explore_speed = float(getattr(options, "explore_speed", 1.0))
        self.v_slide = float(getattr(options, "v_slide", 0.01)) * float(self.explore_speed)
        self.v_fall = float(getattr(options, "v_fall", 0.05)) * float(self.explore_speed)
        self._sm_max_penetration = float(getattr(options, "sm_max_penetration", 0.15))
        self._sm_z_min = float(self.terrain_z_top) - float(max(0.0, self._sm_max_penetration))
        control_mode = str(getattr(options, "control_mode", "teleop")).strip().lower()
        self.auto_state_machine = control_mode == "state-machine"

        # 状态机的关节驱动方式：
        # - ik-position：只写 joint_target_pos（默认，当前行为）
        # - ik-velocity：在写 joint_target_pos 的同时，额外写 joint_target_vel（由两次目标差分得到），更接近“速度指令驱动”
        self.state_machine_drive = str(getattr(options, "state_machine_drive", "ik-position")).strip().lower()
        self._sm_joint_speed_max = float(getattr(options, "sm_joint_speed_max", 6.0))

        # 状态机的关节目标滤波/限速：抑制 IK 解在接触附近的高频抖动。
        # - slew-rate：限制每帧 joint_target_pos 的最大变化（rad/s 量纲）
        # - EMA：对 joint_target_pos / joint_target_vel 做一阶低通
        self._sm_joint_slew_rate = float(getattr(options, "sm_joint_slew_rate", 4.0))
        self._sm_joint_pos_alpha = float(getattr(options, "sm_joint_pos_alpha", 0.15))
        self._sm_joint_vel_alpha = float(getattr(options, "sm_joint_vel_alpha", 0.15))
        self._sm_target_vel_filtered: np.ndarray | None = None
        self.current_state = int(getattr(options, "initial_state", STATE_STEP))
        self._sm_last_state = int(self.current_state)
        self._sm_touch_count = 0
        self._sm_touch_count_th = int(getattr(options, "touch_count_th", 2))
        self._sm_x_init = 0.0
        self._sm_y_init = 0.0
        self._sm_z_touch_origin = 0.0
        self._sm_slide_x_cmd = 0.0
        self._sm_slide_y_cmd = 0.0
        self._sm_slide_z_cmd = 0.0
        self._sm_probe_x = 0.0
        self._sm_probe_y = 0.0
        self._sm_probe_z_origin = 0.0

        # ------------------------------------------------------------------
        # 三角步态轨迹 & 候选点配置
        # ------------------------------------------------------------------
        self.tri_gait_step_length = float(getattr(options, "tri_step_len", 0.15))
        self.tri_gait_step_height = float(getattr(options, "tri_step_h", 0.05))
        self.tri_gait_candidate_num = int(getattr(options, "candidate_num", 5))
        self.candidate_param = str(getattr(options, "candidate_param", "hip-angle")).strip().lower()
        self.tri_gait_traj: list[np.ndarray] = []
        self.candidate_points: list[np.ndarray] = []
        self.candidate_traj_indices: list[int] = []
        self.current_candidate_idx = 0
        self._tri_traj_i = 0
        self._tri_traj_step = 1
        self._sm_lift_target_z: float | None = None
        self._sm_lift_anchor_pos: np.ndarray | None = None
        self._sm_lift_start_z: float | None = None
        self._sm_lift_elapsed = 0.0
        self._sm_lift_max_time = 0.0
        self.stability_check_count = 0
        self.stability_check_frames = 0
        self.stability_check_max_frames = int(getattr(options, "stability_check_max_frames", 10))
        self.stability_check_th = int(getattr(options, "stability_check_th", 7))
        self.x_slide_th = float(getattr(options, "x_slide_th", 0.002))
        self.z_sink_th = float(getattr(options, "z_sink_th", 0.002))
        self.check_press_depth = float(getattr(options, "check_press_depth", 0.05))
        self.check_press_speed = float(getattr(options, "check_press_speed", 0.1)) * float(self.explore_speed)
        self.check_press_done = False
        self.check_press_target: np.ndarray | None = None
        self.check_origin_pos: np.ndarray | None = None
        self.check_press_elapsed = 0.0
        self.check_press_max_time = 0.0
        self.check_retract_speed = float(getattr(options, "check_retract_speed", 0.1)) * float(self.explore_speed)
        self.check_retract_active = False
        self.last_foot_pos: np.ndarray | None = None
        self.check_outward_y = float(getattr(options, "check_outward_y", 0.01))

        # ------------------------------------------------------------------
        # 状态机探索策略（方案二：扇形扫描 + 半径扩张）
        # ------------------------------------------------------------------
        self._sm_explore_distance_init = float(getattr(options, "sm_explore_distance", 0.2))
        self._sm_explore_distance = float(self._sm_explore_distance_init)
        self._sm_explore_distance_step = float(getattr(options, "sm_explore_distance_step", 0.1))
        self._sm_explore_start_angle = float(np.deg2rad(float(getattr(options, "sm_explore_start_angle_deg", -90.0))))
        self._sm_explore_end_angle = float(np.deg2rad(float(getattr(options, "sm_explore_end_angle_deg", 90.0))))
        self._sm_explore_angle_step = float(np.deg2rad(float(getattr(options, "sm_explore_angle_step_deg", 15.0))))
        self._sm_explore_angle_step = float(max(1.0e-6, abs(self._sm_explore_angle_step)))
        self._sm_explore_angle = float(self._sm_explore_start_angle)
        self._sm_lift_height = float(getattr(options, "sm_lift_height", 0.05))

        # 探索圆心（探索失败会抬足回到这里再重新下落）
        self._sm_explore_origin_x = float(self._sm_x_init)
        self._sm_explore_origin_y = float(self._sm_y_init)
        self.is_reliable_area = False
        self.fn_history: list[float] = []
        self.ft_history: list[float] = []
        self.fn_raw_history: list[float] = []
        self.ft_raw_history: list[float] = []
        self.time_history: list[float] = []
        self.enable_plot = bool(getattr(options, "enable_plot", True))
        self._plt = _try_import_matplotlib_pyplot() if self.enable_plot else None
        self.fig = None
        self.ax = None
        self.line_fn = None
        self.line_ft = None
        self.line_fn_filtered = None
        self.line_ft_filtered = None
        if self._plt is None:
            self.enable_plot = False
        if self.enable_plot:
            self._plt.ion()
            self.fig, self.ax = self._plt.subplots()
            (self.line_fn,) = self.ax.plot(
                self.time_history, self.fn_raw_history, label="LF Fn raw", color="C0"
            )
            (self.line_fn_filtered,) = self.ax.plot(
                self.time_history, self.fn_history, label="LF Fn filtered", color="C0", alpha=0.45
            )
            (self.line_ft,) = self.ax.plot(
                self.time_history, self.ft_raw_history, label="LF Ft raw", color="C1"
            )
            (self.line_ft_filtered,) = self.ax.plot(
                self.time_history, self.ft_history, label="LF Ft filtered", color="C1", alpha=0.45
            )
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Force (N)")
            self.ax.set_title("Interaction Force vs. Time")
            self.ax.grid(True)
            self.ax.legend(loc="upper right")


        foot_pos = self.get_foot_position()
        self._sm_x_init = float(foot_pos[0])
        self._sm_y_init = float(foot_pos[1])
        self._sm_z_fall_cmd = float(foot_pos[2])
        self._sm_z_touch_origin = float(foot_pos[2])
        self._sm_slide_x_cmd = float(foot_pos[0])
        self._sm_slide_y_cmd = float(foot_pos[1])
        self._sm_slide_z_cmd = float(foot_pos[2])
        self._sm_probe_x = float(foot_pos[0])
        self._sm_probe_y = float(foot_pos[1])
        self._sm_probe_z_origin = float(foot_pos[2])

        self.tri_gait_start_pos = np.asarray(foot_pos, dtype=np.float64).copy()
        self._generate_tri_gait_traj()
        self._generate_candidate_points()

        # 用真实初始足端位置更新探索圆心
        self._sm_explore_origin_x = float(self._sm_x_init)
        self._sm_explore_origin_y = float(self._sm_y_init)

        self.capture()

    @staticmethod
    def _get_joint_keys(model: newton.Model) -> list[str]:
        keys = list(getattr(model, "joint_key", []))
        return [str(k) for k in keys]

    @classmethod
    def _joint_index_by_name(cls, model: newton.Model, joint_name: str) -> int | None:
        keys = cls._get_joint_keys(model)
        try:
            return int(keys.index(str(joint_name)))
        except ValueError:
            return None

    @classmethod
    def _joint_dof_info_1d(cls, model: newton.Model, joint_name: str) -> tuple[int, float, float] | None:
        """返回 1DoF 关节的 dof 索引与上下限（若不可用则返回 None）。"""

        j = cls._joint_index_by_name(model, joint_name)
        if j is None:
            return None

        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        q0, q1 = int(q_start[j]), int(q_start[j + 1])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        if (d1 - d0) != 1 or (q1 - q0) != 1:
            return None

        lower_np = model.joint_limit_lower.numpy()
        upper_np = model.joint_limit_upper.numpy()
        lower = float(lower_np[q0]) if q0 < lower_np.shape[0] else -np.inf
        upper = float(upper_np[q0]) if q0 < upper_np.shape[0] else np.inf
        return d0, lower, upper

    @classmethod
    def _select_lf_teleop_joints(cls, model: newton.Model) -> list[tuple[str, int, float, float]]:
        """选择 LF 三个可遥操作的 1DoF 关节。

        优先使用常见命名：LF_HAA/LF_HFE/LF_KFE；若不存在则回退到所有以 LF_ 开头的 1DoF 关节前 3 个。
        返回列表元素为 (name, dof_index, lower, upper)。
        """

        preferred = ["LF_HAA", "LF_HFE", "LF_KFE"]
        chosen: list[tuple[str, int, float, float]] = []

        for name in preferred:
            info = cls._joint_dof_info_1d(model, name)
            if info is None:
                continue
            d0, lower, upper = info
            chosen.append((name, int(d0), float(lower), float(upper)))

        if len(chosen) == 3:
            return chosen

        keys = cls._get_joint_keys(model)
        for name in keys:
            if not str(name).startswith("LF_"):
                continue
            if any(name == c[0] for c in chosen):
                continue
            info = cls._joint_dof_info_1d(model, name)
            if info is None:
                continue
            d0, lower, upper = info
            chosen.append((str(name), int(d0), float(lower), float(upper)))
            if len(chosen) == 3:
                break

        if len(chosen) != 3:
            raise ValueError(
                "无法为 LF 选择 3 个 1DoF 关节进行遥操作。"
                "请检查模型 joint_key 是否包含 LF_* 的 1 自由度关节。"
            )
        return chosen

    @staticmethod
    def _cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
        """计算三次 Bezier 曲线在参数 $u$ 处的位置。

        Args:
            p0: 控制点 0，shape=(3,)。
            p1: 控制点 1，shape=(3,)。
            p2: 控制点 2，shape=(3,)。
            p3: 控制点 3，shape=(3,)。
            u: 曲线参数，自动裁剪到 [0, 1]。

        Returns:
            np.ndarray: 曲线点坐标，shape=(3,)。
        """
        u = float(np.clip(u, 0.0, 1.0))
        b0 = (1.0 - u) ** 3
        b1 = 3.0 * (1.0 - u) ** 2 * u
        b2 = 3.0 * (1.0 - u) * u**2
        b3 = u**3
        return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

    def _generate_tri_gait_traj(self) -> None:
        """生成三角步态完整轨迹（三次 Bezier 曲线）。"""
        p0 = np.asarray(self.tri_gait_start_pos, dtype=np.float64)
        p1 = np.array(
            [
                p0[0] + float(self.tri_gait_step_length) / 3.0,
                p0[1],
                p0[2] + float(self.tri_gait_step_height) * 0.5,
            ],
            dtype=np.float64,
        )
        p2 = np.array(
            [
                p0[0] + float(self.tri_gait_step_length) * 2.0 / 3.0,
                p0[1],
                p0[2] + float(self.tri_gait_step_height),
            ],
            dtype=np.float64,
        )
        p3 = np.array(
            [
                p0[0] + float(self.tri_gait_step_length),
                p0[1],
                p0[2],
            ],
            dtype=np.float64,
        )

        self.tri_gait_traj = []
        for u in np.linspace(0.0, 1.0, 100):
            pos = self._cubic_bezier(p0, p1, p2, p3, float(u))
            self.tri_gait_traj.append(np.asarray(pos, dtype=np.float64))

    def _generate_candidate_points(self) -> None:
        """将轨迹离散为候选落足点（等分点，从远到近）。

        支持两种等分参数：
        - `hip-angle`：按 LF 髋关节（默认取 `_lf_teleop_joints[0]`，通常是 LF_HAA）的累计转角等分取点。
        - `xy-distance`：按轨迹索引等分，并按 XY 距离从远到近排序（旧逻辑兜底）。
        """
        if not self.tri_gait_traj:
            self.candidate_points = []
            self.candidate_traj_indices = []
            self.current_candidate_idx = 0
            return

        candidate_num = int(max(1, self.tri_gait_candidate_num))
        start_pos = np.asarray(self.tri_gait_start_pos, dtype=np.float64)
        n = int(len(self.tri_gait_traj))

        idxs: list[int] = []
        method = str(getattr(self, "candidate_param", "hip-angle")).strip().lower()

        if method == "hip-angle":
            try:
                # 用户期望的“髋关节（LF 与 base 连接的电机）”通常就是 LF_HAA。
                _hip_name, hip_dof_idx, _hip_lo, _hip_hi = self._lf_teleop_joints[0]
                d = int(hip_dof_idx)

                leg = next((l for l in self.legs if l.name == "LF"), None)
                if leg is None:
                    raise RuntimeError("未找到 LF 足端配置")

                angles: list[float] = []
                q_seed = self.state_0.joint_q.numpy().astype(np.float64, copy=False)

                for p in self.tri_gait_traj:
                    pos = np.asarray(p, dtype=np.float64)
                    if pos.shape[0] != 3 or (not np.all(np.isfinite(pos))):
                        raise RuntimeError("tri_gait_traj 含非有限点")

                    leg.pos_obj.set_target_position(0, wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])))
                    seed_q = wp.array(self.state_0.joint_q, shape=(1, int(self.model.joint_coord_count)))
                    self._ik_solver.reset()
                    self._ik_solver.step(seed_q, self._ik_solution_q, iterations=self._ik_iters)

                    q_sol = np.asarray(self._ik_solution_q.numpy()[0], dtype=np.float64)
                    if not np.all(np.isfinite(q_sol)):
                        q_sol = q_seed

                    target_pos_dof = self._coord_q_to_target_pos_dof(self.model, q_sol)
                    if not (0 <= d < int(target_pos_dof.shape[0])):
                        raise RuntimeError("hip dof index 越界")
                    a = float(target_pos_dof[d])
                    if not np.isfinite(a):
                        raise RuntimeError("hip angle 非有限")
                    angles.append(a)

                if len(angles) != n:
                    raise RuntimeError("angles 长度不匹配")

                ang = np.unwrap(np.asarray(angles, dtype=np.float64))
                s = np.zeros((n,), dtype=np.float64)
                if n >= 2:
                    s[1:] = np.cumsum(np.abs(np.diff(ang)))

                total = float(s[-1])
                if (not np.isfinite(total)) or total <= 1.0e-10:
                    raise RuntimeError("hip angle 总变化过小")

                samples = np.linspace(0.0, total, candidate_num)
                idxs = [int(np.searchsorted(s, float(v), side="left")) for v in samples]
                idxs = [int(np.clip(i, 0, n - 1)) for i in idxs]
                idxs = sorted(set(int(i) for i in idxs))

                # 从“远到近”：按累计转角弧长从大到小尝试。
                idxs.sort(key=lambda i: float(s[int(i)]), reverse=True)
            except Exception:
                # 回退：使用旧的按轨迹索引等分 + XY 距离排序。
                idxs = []

        if not idxs:
            raw = np.linspace(0, n - 1, candidate_num, dtype=int)
            idxs = sorted(set(int(i) for i in raw.tolist()))
            idxs.sort(
                key=lambda i: float(
                    np.linalg.norm(np.asarray(self.tri_gait_traj[int(i)], dtype=np.float64)[:2] - start_pos[:2])
                ),
                reverse=True,
            )

        self.candidate_traj_indices = [int(i) for i in idxs]
        self.candidate_points = [np.asarray(self.tri_gait_traj[int(i)], dtype=np.float64).copy() for i in idxs]

        self.current_candidate_idx = 0
        self._tri_traj_i = 0

    def _check_stability(self) -> bool:
        """多帧稳定性检测：Fz 阈值 + 无滑动 + 无过度沉陷。"""
        current_foot_pos = self.get_foot_position()
        if self.last_foot_pos is None:
            self.last_foot_pos = np.asarray(current_foot_pos, dtype=np.float64)
            return False

        fz_ok = float(self._fz_filtered) > float(self.Fz_th)
        x_slide = abs(float(current_foot_pos[0]) - float(self.last_foot_pos[0]))
        z_sink = abs(float(current_foot_pos[2]) - float(self.last_foot_pos[2]))
        slide_ok = x_slide < float(self.x_slide_th)
        sink_ok = z_sink < float(self.z_sink_th)

        self.last_foot_pos = np.asarray(current_foot_pos, dtype=np.float64)

        if fz_ok and slide_ok and sink_ok:
            self.stability_check_count += 1
        else:
            self.stability_check_count = 0

        if self.stability_check_count >= int(self.stability_check_th):
            self.stability_check_count = 0
            self.last_foot_pos = None
            return True
        return False

    def _drive_foot_to(self, target_pos: np.ndarray) -> None:
        """驱动足端到目标位置（复用现有 IK 逻辑）。"""
        self.set_foot_position(float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))

    def capture(self) -> None:
        """设置 CUDA graph capture（结构对齐 robot_terrain）。"""
        self.graph = None

        if bool(getattr(self, "_disable_cuda_graph", False)):
            self.sand_graph = None
            return

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_leg()
            self.graph = capture.graph

        self.sand_graph = None
        if wp.get_device().is_cuda and getattr(self.solver_mpm, "grid_type", None) == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_mpm()
            self.sand_graph = capture.graph

    def simulate_leg(self) -> None:
        """刚体仿真：执行 XPBD 动力学与碰撞检测。"""
        # teleop 模式下才使用 settle_time 的“先站稳再走/遥控”。
        # state-machine 模式由 `_state_machine_step()` 统一接管足端目标，不应被 stand/walk 覆盖。
        if not bool(getattr(self, "auto_state_machine", False)):
            if self.sim_time < self.settle_time:
                self.apply_control_stand()
            else:
                self.apply_control_walk()

        for _sub in range(self.sim_substeps):
            self.state_0.clear_forces()

            # 从上一帧（上一轮 simulate_mpm）收集到的冲量估计沙子施加到刚体的力。
            if self._n_colliders > 0 and self.state_0.body_q is not None and getattr(self.state_0, "body_f", None) is not None:
                wp.launch(
                    compute_body_forces,
                    dim=int(self._n_colliders),
                    inputs=[
                        float(self.frame_dt),
                        self.collider_impulse_ids,
                        self.collider_impulses,
                        self.collider_impulse_pos,
                        self.collider_body_id,
                        self.state_0.body_q,
                        self.model.body_com,
                        self.state_0.body_f,
                    ],
                )
                # 保存本子步施加的沙子力，用于后续 subtract_body_force
                self.body_sand_forces.assign(self.state_0.body_f)
            else:
                self.body_sand_forces.zero_()

            self.viewer.apply_forces(self.state_0)

            if self.rigid_solver == "xpbd":
                self.contacts = self.model.collide(self.state_0)
                self.solver_xpbd.step(self.state_0, self.state_1, control=self.control, contacts=self.contacts, dt=self.sim_dt)
            else:
                self.solver_mujoco.step(self.state_0, self.state_1, control=self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.show_foot_traj:
            self._record_lf_rf_foot_positions()

    def simulate_mpm(self) -> None:
        """MPM 仿真：推进粒子并收集冲量/统计足端力。"""
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            wp.launch(
                subtract_body_force,
                dim=self.state_0.body_q.shape,
                inputs=[
                    float(self.frame_dt),
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        # MPM 单步 dt 过大时更容易数值发散（表现为粒子 NaN/Inf 或直接飞出视野）。
        # 默认保持 1（兼容原行为），需要时可通过 --mpm-substeps 增大。
        sub_dt = float(self.frame_dt) / float(max(1, self.mpm_substeps))
        for _ in range(int(self.mpm_substeps)):
            self.solver_mpm.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=sub_dt)
        self.collect_collider_impulses()

        self.foot_particle_force.zero_()
        if self._n_colliders > 0:
            wp.launch(
                compute_single_body_contact_force,
                dim=int(self._n_colliders),
                inputs=[
                    float(self.frame_dt),
                    int(getattr(self, "_foot_contact_body", self.foot)),
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_body_id,
                    self.foot_particle_force,
                ],
            )
            self.foot_particle_force_world = self.foot_particle_force.numpy()[0].astype(np.float32)

        if bool(getattr(self, "_debug_mpm_health", False)):
            self._maybe_debug_mpm_health()

    def step(self) -> None:
        """主仿真步骤：结构对齐 robot_terrain。"""
        self._frame_index += 1
        # 参考 el_mini_move3：外环（每帧）先更新控制目标，再推进物理。
        if self.auto_state_machine:
            self._state_machine_step(self.frame_dt)

        if getattr(self, "graph", None):
            wp.capture_launch(self.graph)
        else:
            self.simulate_leg()

        if getattr(self, "sand_graph", None):
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_mpm()

        self.sim_time += self.frame_dt
        self._update_force_filter()

        fx, fy, fz = self.foot_particle_force_world
        fn = float(self._fz_filtered)
        ft = float(np.sqrt(float(self._fx_filtered * self._fx_filtered + self._fy_filtered * self._fy_filtered)))
        fn_raw = float(max(0.0, float(fz)))
        ft_raw = float(np.sqrt(float(fx * fx + fy * fy)))
        force_magnitude = float(np.sqrt(float(fx * fx + fy * fy + fz * fz)))
        self.fn_history.append(fn)
        self.ft_history.append(ft)
        self.fn_raw_history.append(fn_raw)
        self.ft_raw_history.append(ft_raw)
        self.time_history.append(float(self.sim_time))
        foot_pos = self.get_foot_position()
        z_sink = float(max(0.0, float(self.terrain_z_top) - float(foot_pos[2])))
        if self._sm_last_fz is None:
            fz_rate = 0.0
        else:
            fz_rate = abs(float(self._fz_filtered) - float(self._sm_last_fz)) / float(self.frame_dt)
        self._sm_last_fz = float(self._fz_filtered)

        if self._headless_null_viewer and self.auto_state_machine:
            # 无头调试：定频打印状态机关键量（避免“看不到在探索”）。
            # 用 history 长度当作帧计数（每 ~0.5s 打印一次，默认 fps=60）。
            frame = len(self.time_history)
            if frame % 30 == 0:
                foot_p = self.get_foot_position()
                foot_x = float(foot_p[0])
                foot_y = float(foot_p[1])
                foot_z = float(foot_p[2])
                print(
                    f"t={self.sim_time:7.3f}s frame={frame:5d} state={self.current_state} "
                    f"p=({foot_x:+.3f},{foot_y:+.3f},{foot_z:+.4f}) Fz={self._fz_filtered:+.3f} "
                    f"sink={z_sink:.4f} dFz/dt={fz_rate:.2f}"
                )

        if self._headless_null_viewer and (not bool(getattr(self, "auto_state_machine", False))):
            if force_magnitude > 1e-6:
                self._contact_seen_frames += 1
            else:
                self._contact_seen_frames = 0
            if self._contact_seen_frames >= 3:
                print("\nContact detected; stopping early.")
                raise SystemExit(0)


    def _maybe_debug_mpm_health(self) -> None:
        """轻量级 MPM 健康检查：用于定位粒子“突然消失”的数值原因。"""
        frame = int(getattr(self, "_frame_index", 0))
        if frame <= 0:
            return
        if frame % int(self._debug_mpm_health_every) != 0:
            return

        # 尽量只在采样/低频情况下把数据拉回 host，避免拖慢仿真。
        q = self.sand_state_0.particle_q.numpy()
        qd = self.sand_state_0.particle_qd.numpy() if getattr(self.sand_state_0, "particle_qd", None) is not None else None

        flags = None
        try:
            flags = self.solver_mpm.model.particle_flags.numpy()  # type: ignore[attr-defined]
        except Exception:
            flags = None

        if flags is not None:
            active_mask = (flags.astype(np.int64) & int(newton.ParticleFlags.ACTIVE)) != 0
        else:
            active_mask = np.ones((q.shape[0],), dtype=bool)

        # 可选采样：仅用于快速判断“是否爆炸/NaN”，不用于严格统计。
        if int(self._debug_mpm_health_sample) > 0 and q.shape[0] > int(self._debug_mpm_health_sample):
            rng = np.random.default_rng(seed=frame)
            idx = rng.choice(q.shape[0], size=int(self._debug_mpm_health_sample), replace=False)
            q = q[idx]
            if qd is not None:
                qd = qd[idx]
            active_mask = active_mask[idx]

        q_act = q[active_mask]
        if q_act.size == 0:
            msg = f"[MPM-HEALTH] frame={frame} ACTIVE=0/{q.shape[0]} (all inactive?)"
            print(msg)
            if bool(self._debug_mpm_health_stop):
                raise SystemExit(2)
            return

        finite_q = np.isfinite(q_act).all(axis=1)
        nan_q = int((~finite_q).sum())
        q_finite = q_act[finite_q] if nan_q > 0 else q_act

        if q_finite.size == 0:
            foot_p = self.get_foot_position()
            state = int(getattr(self, "current_state", -1))
            cand = int(getattr(self, "current_candidate_idx", -1))
            tri_i = int(getattr(self, "_tri_traj_i", -1))
            print(
                "[MPM-HEALTH] BAD "
                f"frame={frame} state={state} cand={cand} tri_i={tri_i} "
                f"ACTIVE={int(active_mask.sum())}/{int(active_mask.shape[0])} "
                f"nan_q={nan_q} (all active positions non-finite) "
                f"foot=({float(foot_p[0]):+.3f},{float(foot_p[1]):+.3f},{float(foot_p[2]):+.4f})"
            )
            if bool(self._debug_mpm_health_stop):
                raise SystemExit(2)
            return

        lo = q_finite.min(axis=0)
        hi = q_finite.max(axis=0)
        span = hi - lo

        max_speed = float("nan")
        nan_v = 0
        if qd is not None:
            v_act = qd[active_mask]
            finite_v = np.isfinite(v_act).all(axis=1)
            nan_v = int((~finite_v).sum())
            if finite_v.any():
                max_speed = float(np.linalg.norm(v_act[finite_v], axis=1).max())

        # 触发条件：NaN/Inf、速度爆炸、或 AABB 明显失控。
        bad = False
        if nan_q > 0 or nan_v > 0:
            bad = True
        if (not math.isnan(max_speed)) and max_speed > 50.0:
            bad = True
        if float(span.max()) > 20.0:
            bad = True

        if bad:
            foot_p = self.get_foot_position()
            state = int(getattr(self, "current_state", -1))
            cand = int(getattr(self, "current_candidate_idx", -1))
            tri_i = int(getattr(self, "_tri_traj_i", -1))
            print(
                "[MPM-HEALTH] BAD "
                f"frame={frame} state={state} cand={cand} tri_i={tri_i} "
                f"ACTIVE={int(active_mask.sum())}/{int(active_mask.shape[0])} "
                f"nan_q={nan_q} nan_v={nan_v} max|v|={max_speed:.3f} "
                f"AABB_lo=({lo[0]:+.3f},{lo[1]:+.3f},{lo[2]:+.3f}) "
                f"AABB_hi=({hi[0]:+.3f},{hi[1]:+.3f},{hi[2]:+.3f}) "
                f"foot=({float(foot_p[0]):+.3f},{float(foot_p[1]):+.3f},{float(foot_p[2]):+.4f})"
            )
            if bool(self._debug_mpm_health_stop):
                raise SystemExit(2)
        else:
            # 低频简报：便于确认粒子是不是“整体飞走/整体下沉”。
            if self._headless_null_viewer:
                print(
                    "[MPM-HEALTH] "
                    f"frame={frame} ACTIVE={int(active_mask.sum())}/{int(active_mask.shape[0])} "
                    f"max|v|={max_speed:.3f} span=({span[0]:.3f},{span[1]:.3f},{span[2]:.3f})"
                )


    def _record_lf_rf_foot_positions(self) -> None:
        """记录 LF/RF 足端点的世界坐标轨迹（用于 viewer 折线 debug）。"""
        body_q_np = self.state_0.body_q.numpy()

        lf = next((l for l in self.legs if l.name == "LF"), None)
        rf = next((l for l in self.legs if l.name == "RF"), None)
        if lf is not None:
            self._traj_lf.append(self._foot_point_world(body_q_np, lf))
        if rf is not None:
            self._traj_rf.append(self._foot_point_world(body_q_np, rf))

    def _log_polyline(self, path: str, pts, color) -> None:
        """在 viewer 中绘制一条折线（按相邻点连线）。

        Args:
            path: viewer 的日志路径。
            pts: 点序列（可迭代，每个元素可转为 shape=(3,)）。
            color: 颜色（wp.vec3）。
        """
        if len(pts) < 2:
            self.viewer.log_lines(path, None, None, None)
            return

        p = np.asarray(list(pts), dtype=np.float32)
        starts_np = p[:-1]
        ends_np = p[1:]
        starts = wp.array(starts_np, dtype=wp.vec3, device=self.model.device)
        ends = wp.array(ends_np, dtype=wp.vec3, device=self.model.device)
        colors = wp.full(starts.shape[0], value=color, dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines(path, starts=starts, ends=ends, colors=colors)

    def apply_control_walk(self) -> None:
        """遥操作外环：控制 LF 三关节目标并写入控制缓冲。

        键位（关节目标增量）：
            - `i`/`k`: 关节 0  + / -
            - `j`/`l`: 关节 1  + / -
            - `u`/`o`: 关节 2  + / -

        说明：
            - 允许同时按多个方向键，三个关节的变化彼此独立叠加。
            - 无按键时不更新目标（保持上一目标，确保关节保持刚度）。
            - 非 LF 关节目标始终保持初始化站立值不变。
        """

        def _key_down(k: str) -> bool:
            if not hasattr(self.viewer, "is_key_down"):
                return False
            return bool(self.viewer.is_key_down(k))

        dirs = np.array(
            [float(_key_down("i")) - float(_key_down("k")), float(_key_down("j")) - float(_key_down("l")), float(_key_down("u")) - float(_key_down("o"))],
            dtype=np.float64,
        )

        if float(np.linalg.norm(dirs, ord=1)) > 0.0:
            dq = dirs * float(self._teleop_joint_speed) * float(self.frame_dt)
            for joint_i, (_name, dof_index, lower, upper) in enumerate(self._lf_teleop_joints):
                d = int(dof_index)
                self._target_pos_dof_cache[d] = float(self._target_pos_dof_cache[d]) + float(dq[joint_i])
                if np.isfinite(lower) or np.isfinite(upper):
                    self._target_pos_dof_cache[d] = float(np.clip(self._target_pos_dof_cache[d], lower, upper))

        self._set_control_targets(target_pos_dof=self._target_pos_dof_cache, target_vel_dof=None)

    def apply_control_stand(self) -> None:
        """站立外环：所有关节保持初始化站立目标（含 LF）。"""

        self._set_control_targets(target_pos_dof=self._target_pos_dof_cache, target_vel_dof=None)

    def _solve_ik_and_write_targets(self) -> None:
        """执行一次 IK 求解，并把解写入关节目标缓冲区。"""
        seed_q = wp.array(self.state_0.joint_q, shape=(1, int(self.model.joint_coord_count)))
        self._ik_solver.reset()
        self._ik_solver.step(seed_q, self._ik_solution_q, iterations=self._ik_iters)

        q_sol = np.asarray(self._ik_solution_q.numpy()[0], dtype=np.float64)
        target_pos_dof = self._coord_q_to_target_pos_dof(self.model, q_sol)
        self._set_control_targets(target_pos_dof=target_pos_dof, target_vel_dof=None)

    def _set_control_targets(self, *, target_pos_dof: np.ndarray, target_vel_dof: np.ndarray | None) -> None:
        """把 dof-space 的目标位置/速度写入 `self.control`。

        Args:
            target_pos_dof: 目标关节位置（dof-space），shape=(joint_dof_count,)。
            target_vel_dof: 目标关节速度（dof-space）。为 None 时不更新速度目标。
        """
        target_pos = np.asarray(target_pos_dof, dtype=np.float64)
        pos_wp = wp.array(target_pos.astype(np.float32), dtype=self._control_dtype_pos, device=self.model.device)
        wp.copy(self.control.joint_target_pos, pos_wp)

        if target_vel_dof is not None:
            target_vel = np.asarray(target_vel_dof, dtype=np.float64)
            vel_wp = wp.array(target_vel.astype(np.float32), dtype=self._control_dtype_vel, device=self.model.device)
            wp.copy(self.control.joint_target_vel, vel_wp)

    def get_foot_raw_data(self) -> tuple[np.ndarray, np.ndarray]:
        """读取足端原始力与位置。

        Returns:
            tuple[np.ndarray, np.ndarray]: (force_xyz, position_xyz)，单位分别为 N 与 m。
        """
        force = np.asarray(self.foot_particle_force_world, dtype=np.float32).copy()

        if self.state_0.body_q is None:
            position = np.zeros(3, dtype=np.float64)
            return force, position

        leg = next((l for l in self.legs if l.name == "LF"), None)
        if leg is not None:
            position = self._foot_point_world(self.state_0.body_q.numpy(), leg)
        else:
            tf = self.state_0.body_q.numpy()[int(self.foot)]
            position = np.asarray(tf[:3], dtype=np.float64)

        return force, position

    def set_foot_position(self, x_target: float, y_target: float, z_target: float) -> None:
        """控制足端绝对位置（对接轨迹规划 + IK)。

        Args:
            x_target: 目标世界坐标 x。
            y_target: 目标世界坐标 y。
            z_target: 目标世界坐标 z。
        """
        if not (np.isfinite(x_target) and np.isfinite(y_target) and np.isfinite(z_target)):
            return

        leg = next((l for l in self.legs if l.name == "LF"), None)
        if leg is None:
            raise RuntimeError("未找到 LF 足端配置，无法设置目标位置。")

        # 参考 el_mini_move3：直接更新 IK objective 的 target，避免每次分配 wp.array。
        leg.pos_obj.set_target_position(
            0,
            wp.vec3(float(x_target), float(y_target), float(z_target)),
        )

        # 先尝试 Newton IK；如果它给出“不变解”但任务空间误差明显，则启用数值 IK 兜底。
        current_pos = self.get_foot_position()
        target_pos = np.array([float(x_target), float(y_target), float(z_target)], dtype=np.float64)
        pos_err = float(np.linalg.norm(target_pos - np.asarray(current_pos, dtype=np.float64)))

        seed_q = wp.array(self.state_0.joint_q, shape=(1, int(self.model.joint_coord_count)))
        self._ik_solver.reset()
        self._ik_solver.step(seed_q, self._ik_solution_q, iterations=self._ik_iters)

        q_sol = np.asarray(self._ik_solution_q.numpy()[0], dtype=np.float64)
        q_seed = self.state_0.joint_q.numpy().astype(np.float64, copy=False)

        if not np.all(np.isfinite(q_sol)):
            q_sol = q_seed.copy()

        def _fk_foot_point_for_q(q_vec: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(q_vec)):
                return np.full(3, np.nan, dtype=np.float64)
            self._ik_numeric_state.joint_q.assign([float(x) for x in q_vec])
            if getattr(self._ik_numeric_state, "joint_qd", None) is not None:
                self._ik_numeric_state.joint_qd.zero_()
            newton.eval_fk(self.model, self._ik_numeric_state.joint_q, self._ik_numeric_state.joint_qd, self._ik_numeric_state)
            body_q_np = self._ik_numeric_state.body_q.numpy()
            return self._foot_point_world(body_q_np, leg).astype(np.float64)

        # Newton IK 在一些链路上可能出现“解几乎不动”，尤其当目标每帧只变化很小。
        # 只要任务空间误差仍然明显且 Newton IK 没有把误差显著降下来，就切到数值 IK 兜底。
        q_delta = float(np.linalg.norm(q_sol - q_seed)) if np.all(np.isfinite(q_sol)) else float("inf")
        use_numeric_ik = False
        if pos_err > 1.0e-4 and q_delta < 1.0e-6:
            use_numeric_ik = True
        elif pos_err > 1.0e-3:
            p_sol = _fk_foot_point_for_q(q_sol)
            if (not np.all(np.isfinite(p_sol))) or (float(np.linalg.norm(target_pos - p_sol)) > 0.9 * float(pos_err)):
                use_numeric_ik = True

        if use_numeric_ik:
            q_sol = self._solve_ik_numeric_position(target_pos)

        if not np.all(np.isfinite(q_sol)):
            q_sol = q_seed.copy()

        target_pos_dof = self._coord_q_to_target_pos_dof(self.model, q_sol)

        # 防内收：在 dof-space 上对髋外展/内收自由度做限幅（dof 索引与 target_pos_dof 一致）。
        # 只约束 HAA，避免过度裁剪导致足端无法下压到目标深度。
        try:
            _haa_name, haa_dof_idx, _haa_lower, _haa_upper = self._lf_teleop_joints[0]
            d = int(haa_dof_idx)
            if 0 <= d < int(target_pos_dof.shape[0]):
                target_pos_dof[d] = float(np.clip(float(target_pos_dof[d]), 0.0, float(np.deg2rad(30.0))))
        except Exception:
            pass

        # state-machine 模式下：对 joint_target_pos 做限速与低通滤波，抑制接触导致的 IK 解抖动。
        target_pos_cmd = np.asarray(target_pos_dof, dtype=np.float64)
        target_vel_dof = None

        if bool(getattr(self, "auto_state_machine", False)):
            prev = np.asarray(getattr(self, "_target_pos_dof_cache", target_pos_cmd), dtype=np.float64)
            dt = float(max(1.0e-8, getattr(self, "frame_dt", 0.0)))

            # CHECK 下压 / LIFT / BACKTRACK 阶段需要更强的响应：
            # - CHECK 下压：若 slew/EMA 太强，会导致“命令下压很慢”，拖慢仿真；
            # - LIFT/BACKTRACK：若 slew/EMA 太强，会导致足端难以从粒子里拔出。
            # 因此在这些阶段临时禁用 slew/EMA，直接下发 IK 解。
            state = int(getattr(self, "current_state", 0))
            pressing = state == STATE_CHECK and (not bool(getattr(self, "check_press_done", False)))
            aggressive = state in (STATE_LIFT, STATE_BACKTRACK) or pressing
            if aggressive:
                slew_rate = 0.0
                alpha_p = 0.0
            else:
                slew_rate = float(getattr(self, "_sm_joint_slew_rate", 0.0))
                alpha_p = float(np.clip(float(getattr(self, "_sm_joint_pos_alpha", 0.0)), 0.0, 1.0))
            if slew_rate > 0.0:
                delta_max = float(slew_rate) * dt
                delta = target_pos_cmd - prev
                target_pos_cmd = prev + np.clip(delta, -delta_max, delta_max)
            if alpha_p > 0.0:
                target_pos_cmd = (1.0 - alpha_p) * prev + alpha_p * target_pos_cmd

            if str(getattr(self, "state_machine_drive", "ik-position")) == "ik-velocity":
                v = (target_pos_cmd - prev) / dt
                v_max = float(max(0.0, getattr(self, "_sm_joint_speed_max", 0.0)))
                if v_max > 0.0:
                    v = np.clip(v, -v_max, v_max)

                # 速度目标再做一次低通，减少速度指令的高频抖动。
                alpha_v = float(np.clip(float(getattr(self, "_sm_joint_vel_alpha", 0.0)), 0.0, 1.0))
                v_f = getattr(self, "_sm_target_vel_filtered", None)
                if v_f is None:
                    v_f = np.zeros_like(v)
                v_f = (1.0 - alpha_v) * np.asarray(v_f, dtype=np.float64) + alpha_v * np.asarray(v, dtype=np.float64)
                self._sm_target_vel_filtered = v_f
                target_vel_dof = v_f

        # 注意：MuJoCo 的 position actuator 只看 joint_target_pos；因此这里保持 joint_target_pos 始终写入，确保两种求解器都可用。
        self._target_pos_dof_cache = np.asarray(target_pos_cmd, dtype=np.float64).copy()
        self._set_control_targets(target_pos_dof=self._target_pos_dof_cache, target_vel_dof=target_vel_dof)

    def _solve_ik_numeric_position(self, target_pos: np.ndarray) -> np.ndarray:
        """数值 IK 兜底：用有限差分雅可比把 LF 足端点推到目标位置。

        Args:
            target_pos: 世界系目标位置，shape=(3,)。

        Returns:
            np.ndarray: joint coord-space 解向量，shape=(joint_coord_count,)。
        """

        leg = next((l for l in self.legs if l.name == "LF"), None)
        if leg is None:
            return self.state_0.joint_q.numpy().astype(np.float64)

        q = self.state_0.joint_q.numpy().astype(np.float64, copy=True)
        q_seed = q.copy()
        eps = float(max(1.0e-6, self._ik_numeric_eps))
        lam = float(max(0.0, self._ik_numeric_damping))

        q_lo = None
        q_hi = None
        if getattr(self.model, "joint_limit_lower", None) is not None and getattr(self.model, "joint_limit_upper", None) is not None:
            try:
                q_lo = self.model.joint_limit_lower.numpy().astype(np.float64, copy=False)
                q_hi = self.model.joint_limit_upper.numpy().astype(np.float64, copy=False)
            except Exception:
                q_lo = None
                q_hi = None

        def _fk_point(q_vec: np.ndarray) -> np.ndarray:
            if not np.all(np.isfinite(q_vec)):
                return np.full(3, np.nan, dtype=np.float64)

            self._ik_numeric_state.joint_q.assign([float(x) for x in q_vec])
            if getattr(self._ik_numeric_state, "joint_qd", None) is not None:
                self._ik_numeric_state.joint_qd.zero_()
            newton.eval_fk(self.model, self._ik_numeric_state.joint_q, self._ik_numeric_state.joint_qd, self._ik_numeric_state)
            body_q_np = self._ik_numeric_state.body_q.numpy()
            return self._foot_point_world(body_q_np, leg).astype(np.float64)

        for _ in range(int(max(1, self._ik_numeric_max_iters))):
            p = _fk_point(q)
            if not np.all(np.isfinite(p)):
                return q_seed
            e = target_pos - p
            if float(np.linalg.norm(e)) < 5.0e-5:
                break

            n = int(q.shape[0])
            J = np.zeros((3, n), dtype=np.float64)
            for j in range(n):
                q_pert = q.copy()
                q_pert[j] += eps
                p_pert = _fk_point(q_pert)
                if not np.all(np.isfinite(p_pert)):
                    return q_seed
                J[:, j] = (p_pert - p) / eps

            # Damped least squares: dq = J^T (J J^T + lam I)^-1 e
            A = J @ J.T
            if lam > 0.0:
                A = A + lam * np.eye(3, dtype=np.float64)
            try:
                y = np.linalg.solve(A, e)
            except np.linalg.LinAlgError:
                y = np.linalg.lstsq(A, e, rcond=None)[0]
            dq = (J.T @ y)

            if not np.all(np.isfinite(dq)):
                return q_seed

            # 限制单次更新幅度，避免数值发散
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 0.25:
                dq = dq * (0.25 / dq_norm)

            q = q + dq
            if not np.all(np.isfinite(q)):
                return q_seed
            if q_lo is not None and q_hi is not None:
                q = np.clip(q, q_lo, q_hi)

        return q

    def get_foot_position(self) -> np.ndarray:
        """返回足端的世界坐标位置。"""
        _force, position = self.get_foot_raw_data()
        return position

    def _update_force_filter(self) -> None:
        """更新足端法向力的滤波值。"""
        fx_raw, fy_raw, fz_raw = self.foot_particle_force_world
        alpha = float(np.clip(self._f_filter_alpha, 0.0, 1.0))

        fx = (1.0 - alpha) * float(self._fx_filtered) + alpha * float(fx_raw)
        fy = (1.0 - alpha) * float(self._fy_filtered) + alpha * float(fy_raw)
        fz = (1.0 - alpha) * float(self._fz_filtered) + alpha * float(fz_raw)

        f_max = float(max(0.0, self._f_max))
        fx = float(np.clip(fx, -f_max, f_max))
        fy = float(np.clip(fy, -f_max, f_max))
        fz = float(np.clip(fz, 0.0, f_max))

        self._fx_filtered = fx
        self._fy_filtered = fy
        self._fz_filtered = fz

    def _state_machine_step(self, dt: float) -> None:
        """状态机主逻辑：先沿三角步态前进→落地检测→沿原轨迹回退→重复→锁定。"""
        if not self.candidate_points:
            self._generate_tri_gait_traj()
            self._generate_candidate_points()
            if not self.candidate_points:
                return

        if not self.candidate_traj_indices:
            self._generate_candidate_points()
            if not self.candidate_traj_indices:
                return

        foot_pos = self.get_foot_position()
        if not np.all(np.isfinite(np.asarray(foot_pos, dtype=np.float64))) or not np.isfinite(float(self._fz_filtered)):
            self.current_state = STATE_STEP
            self.stability_check_count = 0
            self.stability_check_frames = 0
            self.last_foot_pos = None
            return

        if int(self.current_state) != int(getattr(self, "_sm_last_state", self.current_state)):
            self._sm_last_state = int(self.current_state)
            cand_n = int(len(getattr(self, "candidate_traj_indices", [])))
            cand_i = int(getattr(self, "current_candidate_idx", 0))
            tri_i = int(getattr(self, "_tri_traj_i", 0))
            print(
                f"\n[SM] enter state={self.current_state} t={self.sim_time:.3f}s "
                f"cand={cand_i}/{cand_n} tri_i={tri_i}"
            )

        if self.current_state == STATE_STEP:
            # 先按三角步态轨迹把脚跨到“最远候选点”（通常是轨迹末端）。
            target_i = int(self.candidate_traj_indices[int(self.current_candidate_idx)])
            target_i = int(np.clip(target_i, 0, len(self.tri_gait_traj) - 1))

            if int(getattr(self, "_tri_traj_i", 0)) < target_i:
                step = int(max(1, getattr(self, "_tri_traj_step", 2)))
                self._tri_traj_i = int(min(int(getattr(self, "_tri_traj_i", 0)) + step, target_i))
                pos = np.asarray(self.tri_gait_traj[int(self._tri_traj_i)], dtype=np.float64)
                self._drive_foot_to(pos)
                return

            # 到达最远点：进入落地检测。
            self.current_state = STATE_CHECK
            self.stability_check_frames = 0
            self.stability_check_count = 0
            self.last_foot_pos = None
            self.check_press_done = False
            self.check_retract_active = False
            self.check_press_elapsed = 0.0
            self.check_press_max_time = 0.0
            # 下压检测时锁定“当前足端世界系 XY”，避免追逐预定义轨迹点导致的水平晃动/内收。
            # 这样在基座/髋相对运动时，腿会自然向后回摆；下压过程仅改变 Z。
            self.check_origin_pos = np.asarray(self.get_foot_position(), dtype=np.float64).copy()
            outward = float(max(0.0, getattr(self, "check_outward_y", 0.0)))
            if outward > 0.0:
                side_sign = 1.0 if float(self.check_origin_pos[1]) >= 0.0 else -1.0
                self.check_origin_pos[1] = float(self.check_origin_pos[1]) + side_sign * outward
            self.check_press_target = None
            self._sm_lift_target_z = None
            self._sm_lift_anchor_pos = None
            self._sm_lift_start_z = None
            self._sm_lift_elapsed = 0.0
            self._sm_lift_max_time = 0.0

        elif self.current_state == STATE_CHECK:
            # CHECK 分三段：落地(按速度下压) -> 触地后保持并多帧判定 -> 失败则抬回并进入回退。
            if self.check_origin_pos is None:
                self.check_origin_pos = np.asarray(self.get_foot_position(), dtype=np.float64).copy()
                outward = float(max(0.0, getattr(self, "check_outward_y", 0.0)))
                if outward > 0.0:
                    side_sign = 1.0 if float(self.check_origin_pos[1]) >= 0.0 else -1.0
                    self.check_origin_pos[1] = float(self.check_origin_pos[1]) + side_sign * outward

            if self.check_press_target is None:
                origin = np.asarray(self.check_origin_pos, dtype=np.float64)
                max_pen = float(max(0.0, getattr(self, "_sm_max_penetration", 0.0)))
                press_depth = float(max(0.0, getattr(self, "check_press_depth", 0.0)))
                if max_pen > 0.0:
                    press_depth = float(min(press_depth, max_pen))
                z_min = float(getattr(self, "_sm_z_min", origin[2]))
                # 让“下压深度”与地形顶面绑定（terrain_z_top），避免进入 CHECK 时脚抬得较高导致压入不足。
                # 仍由 z_min (= terrain_z_top - sm_max_penetration) 做硬下限，防止过深引发数值发散。
                if press_depth > 0.0:
                    z_target_from_top = float(self.terrain_z_top) - float(press_depth)
                    z_target = float(max(z_target_from_top, z_min))
                    z_target = float(min(float(origin[2]), z_target))
                else:
                    z_target = float(min(float(origin[2]), z_min))
                self.check_press_target = origin.copy()
                self.check_press_target[2] = float(z_target)

                press_speed = float(max(0.0, getattr(self, "check_press_speed", 0.0)))
                down_dist = float(max(0.0, float(origin[2]) - float(z_target)))
                self.check_press_elapsed = 0.0
                if press_speed > 0.0 and down_dist > 0.0:
                    # 经验上 IK/关节限速会让实际下压速度显著低于命令速度，
                    # 若超时过短会导致“还没压到目标深度就结束下压”，表现为与粒子几乎无交互。
                    self.check_press_max_time = float(5.0 * down_dist / press_speed)
                else:
                    self.check_press_max_time = 0.0

            if not bool(self.check_press_done):
                current_foot_pos = self.get_foot_position()
                press_speed = float(max(0.0, getattr(self, "check_press_speed", 0.0)))
                dz = press_speed * float(max(0.0, dt))
                target_z = float(self.check_press_target[2])

                # 一旦检测到足地力满足阈值，认为“落地完成”，停止继续下压，转入保持判定。
                if float(self._fz_filtered) > float(self.Fz_th):
                    hold_pos = np.asarray(self.check_origin_pos, dtype=np.float64).copy()
                    hold_pos[2] = float(current_foot_pos[2])
                    self.check_press_target = hold_pos
                    self.check_press_done = True
                    self.stability_check_frames = 0
                    self.stability_check_count = 0
                    self.last_foot_pos = None
                    return

                new_z = float(max(target_z, float(current_foot_pos[2]) - dz))
                press_cmd = np.asarray(self.check_origin_pos, dtype=np.float64).copy()
                press_cmd[2] = new_z
                self._drive_foot_to(press_cmd)
                press_tol = 1.0e-4
                self.check_press_elapsed += float(max(0.0, dt))
                if float(current_foot_pos[2]) <= target_z + press_tol or self.check_press_elapsed >= float(self.check_press_max_time):
                    self.check_press_done = True
                    self.stability_check_frames = 0
                    self.stability_check_count = 0
                    self.last_foot_pos = None
                return

            self._drive_foot_to(self.check_press_target)
            self.stability_check_frames += 1
            if self._check_stability():
                self.current_state = STATE_LOCK
            elif self.stability_check_frames >= int(self.stability_check_max_frames):
                # 判定失败：直接进入回退，让 BACKTRACK 沿“原三角步态轨迹”回到更近候选点。
                self.current_candidate_idx += 1
                if self.current_candidate_idx >= len(self.candidate_traj_indices):
                    self.current_candidate_idx = max(0, len(self.candidate_traj_indices) - 1)
                    self.current_state = STATE_LOCK
                    return
                self.check_press_target = None
                # 失败抬脚/回退阶段需要用到下压前的锚点（尤其是 z 高度），
                # 不能在这里清空，否则可能导致回退时足端仍在粒子中。
                self.check_press_done = False
                self.check_retract_active = False
                self.last_foot_pos = None
                self.stability_check_frames = 0
                self.stability_check_count = 0
                self._sm_lift_target_z = None
                self._sm_lift_anchor_pos = None
                self._sm_lift_start_z = None
                self._sm_lift_elapsed = 0.0
                self._sm_lift_max_time = 0.0
                self.current_state = STATE_LIFT
                return

        elif self.current_state == STATE_LIFT:
            lift_h = float(max(0.0, getattr(self, "_sm_lift_height", 0.0)))
            if self.check_origin_pos is None:
                self.check_origin_pos = np.asarray(self.get_foot_position(), dtype=np.float64).copy()
                outward = float(max(0.0, getattr(self, "check_outward_y", 0.0)))
                if outward > 0.0:
                    side_sign = 1.0 if float(self.check_origin_pos[1]) >= 0.0 else -1.0
                    self.check_origin_pos[1] = float(self.check_origin_pos[1]) + side_sign * outward
            current_foot_pos = self.get_foot_position()

            # 进入 LIFT 时锁定抬脚锚点：使用“当前足端世界系 XY”。
            # 这样即使基座在 CHECK 期间发生移动，也能保证抬脚任务可达，从而先抬离粒子再回退。
            if self._sm_lift_anchor_pos is None:
                self._sm_lift_anchor_pos = np.asarray(current_foot_pos, dtype=np.float64).copy()
                self._sm_lift_start_z = float(current_foot_pos[2])

            # 进入 LIFT 时锁定一个固定的抬脚目标高度：
            # 1) 至少回到“下压前高度”（check_origin_pos.z），保证回退时足端在粒子上方；
            # 2) 同时确保高于 terrain_z_top + lift_h（把 lift_h 当作最小安全间隙）。
            if self._sm_lift_target_z is None:
                origin_z = float(self.check_origin_pos[2]) if self.check_origin_pos is not None else float(current_foot_pos[2])
                safe_z = float(self.terrain_z_top) + float(lift_h)
                target_z = float(max(origin_z, safe_z, float(current_foot_pos[2])))
                self._sm_lift_target_z = float(target_z)

            lift_speed = float(max(0.0, getattr(self, "check_retract_speed", 0.0)))
            if self._sm_lift_elapsed == 0.0:
                if lift_speed > 0.0:
                    start_z = float(self._sm_lift_start_z) if self._sm_lift_start_z is not None else float(current_foot_pos[2])
                    up_dist = float(max(0.0, float(self._sm_lift_target_z) - start_z))
                    # 类似下压：IK/关节限速会让实际速度慢于命令速度，因此超时要更宽松。
                    self._sm_lift_max_time = float(max(0.0, 5.0 * up_dist / lift_speed))
                else:
                    self._sm_lift_max_time = 0.0
            target_z = float(self._sm_lift_target_z)
            lift_cmd = np.asarray(self._sm_lift_anchor_pos, dtype=np.float64).copy()
            lift_cmd[2] = float(target_z)
            self._drive_foot_to(lift_cmd)
            lift_tol = 1.0e-4
            self._sm_lift_elapsed += float(max(0.0, dt))
            # 只有当足端已经抬到粒子顶面之上（带安全间隙）才允许进入回退，避免在粒子内回退。
            min_backtrack_z = float(self.terrain_z_top) + float(lift_h)
            reached_clearance = float(current_foot_pos[2]) >= float(min_backtrack_z) - lift_tol
            reached_target = float(current_foot_pos[2]) >= target_z - lift_tol
            timed_out = self._sm_lift_elapsed >= float(self._sm_lift_max_time) and float(self._sm_lift_max_time) > 0.0
            # 兜底：只要 LIFT 已经至少下发过一次“安全高度”目标，就允许进入 BACKTRACK。
            # BACKTRACK 本身会持续强制 z>=min_backtrack_z，因此回退过程中足端不会再插在粒子里。
            issued_one_lift_cmd = self._sm_lift_elapsed >= float(max(0.0, dt))
            if reached_target or (timed_out and reached_clearance) or issued_one_lift_cmd:
                self.current_state = STATE_BACKTRACK
            return

        elif self.current_state == STATE_BACKTRACK:
            # 沿“刚才跨步的同一条轨迹”回退到更近的候选点（轨迹等分点）。
            if self.current_candidate_idx >= len(self.candidate_traj_indices):
                self.current_candidate_idx = max(0, len(self.candidate_traj_indices) - 1)
                self.current_state = STATE_LOCK
                return

            target_i = int(self.candidate_traj_indices[int(self.current_candidate_idx)])
            target_i = int(np.clip(target_i, 0, len(self.tri_gait_traj) - 1))

            if int(getattr(self, "_tri_traj_i", 0)) > target_i:
                step = int(max(1, getattr(self, "_tri_traj_step", 2)))
                self._tri_traj_i = int(max(int(getattr(self, "_tri_traj_i", 0)) - step, target_i))
                pos = np.asarray(self.tri_gait_traj[int(self._tri_traj_i)], dtype=np.float64)
                min_backtrack_z = float(self.terrain_z_top) + float(max(0.0, getattr(self, "_sm_lift_height", 0.0)))
                if self._sm_lift_target_z is not None:
                    pos[2] = float(max(float(self._sm_lift_target_z), min_backtrack_z))
                else:
                    pos[2] = float(max(float(pos[2]), min_backtrack_z))
                self._drive_foot_to(pos)
                return

            # 到达更近候选点：重新落地检测。
            self.current_state = STATE_CHECK
            self.stability_check_frames = 0
            self.stability_check_count = 0
            self.last_foot_pos = None
            self.check_press_done = False
            self.check_retract_active = False
            self.check_press_target = None
            self.check_origin_pos = np.asarray(self.get_foot_position(), dtype=np.float64).copy()
            outward = float(max(0.0, getattr(self, "check_outward_y", 0.0)))
            if outward > 0.0:
                side_sign = 1.0 if float(self.check_origin_pos[1]) >= 0.0 else -1.0
                self.check_origin_pos[1] = float(self.check_origin_pos[1]) + side_sign * outward
            self.check_press_elapsed = 0.0
            self.check_press_max_time = 0.0
            self._sm_lift_target_z = None
            self._sm_lift_anchor_pos = None
            self._sm_lift_start_z = None
            self._sm_lift_elapsed = 0.0
            self._sm_lift_max_time = 0.0

        elif self.current_state == STATE_LOCK:
            if self.check_press_target is not None:
                self._drive_foot_to(self.check_press_target)
            else:
                pos = np.asarray(self.tri_gait_traj[int(self._tri_traj_i)], dtype=np.float64)
                self._drive_foot_to(pos)

    @staticmethod
    def _find_free_joint_q0(model: newton.Model) -> int | None:
        """返回 free joint 在 joint_q 中的起始索引 q0；如果没有 free joint 则返回 None。"""

        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        joint_count = int(qd_start.shape[0] - 1)

        for j in range(joint_count):
            q0, q1 = int(q_start[j]), int(q_start[j + 1])
            d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
            if (q1 - q0) == 7 and (d1 - d0) == 6:
                return q0
        return None

    def _get_base_pose_from_state(self, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
        """从状态里取统一参考系（free joint）的世界位姿。

        如果没有 free joint（固定基座），则退化为世界系原点 + 单位四元数。
        """

        if self._free_joint_q0 is None:
            return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        q = state.joint_q.numpy().astype(np.float64)
        q0 = int(self._free_joint_q0)
        base_p = q[q0 : q0 + 3]
        base_q = q[q0 + 3 : q0 + 7]
        return base_p, base_q

    @staticmethod
    def _find_body_index(model: newton.Model, body_name: str) -> int:
        """根据 body 名称在 `model.body_key` 中查找索引。

        Args:
            model: Newton 模型。
            body_name: body 名称（例如 "RF_HIP"）。

        Returns:
            int: body 索引。

        Raises:
            AttributeError: model 不含 body_key。
            ValueError: 未找到对应 body。
        """
        if not hasattr(model, "body_key"):
            raise AttributeError("model 没有 body_key 属性")
        keys = [str(k) for k in list(model.body_key)]
        try:
            return int(keys.index(str(body_name)))
        except ValueError as exc:
            preview = ", ".join(keys[:20])
            raise ValueError(f"找不到 body：'{body_name}'。前 20 个 body_key: [{preview}]") from exc

    @staticmethod
    def _find_body_index_optional(model: newton.Model, body_name: str) -> int | None:
        """可选地根据 body 名称查找索引；找不到返回 None。"""
        if not hasattr(model, "body_key"):
            return None
        keys = [str(k) for k in list(model.body_key)]
        try:
            return int(keys.index(str(body_name)))
        except ValueError:
            return None

    @staticmethod
    def _urdf_joint_origin_xyz(urdf_path: str, *, parent_link: str, child_link: str) -> np.ndarray:
        """从 URDF 里读取 parent->child 关节的 origin xyz（用于 fixed joint 折叠后的足端点）。

        Args:
            urdf_path: URDF 文件路径。
            parent_link: parent link 名称（例如 "LF_SHANK"）。
            child_link: child link 名称（例如 "LF_FOOT"）。

        Returns:
            np.ndarray: shape=(3,) 的 xyz（parent link frame）。找不到则返回 0。
        """

        try:
            root = ET.parse(urdf_path).getroot()
        except Exception:
            return np.zeros(3, dtype=np.float64)

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            if str(parent.get("link", "")) != str(parent_link):
                continue
            if str(child.get("link", "")) != str(child_link):
                continue

            origin = joint.find("origin")
            if origin is None:
                return np.zeros(3, dtype=np.float64)
            xyz_str = origin.get("xyz", "0 0 0")
            try:
                xyz = np.fromstring(xyz_str, sep=" ", dtype=np.float64)
                if xyz.shape[0] != 3:
                    return np.zeros(3, dtype=np.float64)
                return xyz
            except Exception:
                return np.zeros(3, dtype=np.float64)

        return np.zeros(3, dtype=np.float64)

    def _foot_point_world(self, body_q_np: np.ndarray, leg: _LegSpec) -> np.ndarray:
        """返回“足端点”的世界坐标。

        当 *_FOOT body 存在时，offset 为 0；当 *_FOOT 被折叠进 *_SHANK 时，offset 来自 URDF fixed joint origin。
        """

        tf = body_q_np[int(leg.foot_body)]
        p = tf[:3].astype(np.float64)
        q = tf[3:7].astype(np.float64)
        if leg.foot_offset_local is None:
            return p
        off = np.asarray(leg.foot_offset_local, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(off)) < 1e-12:
            return p
        return p + self._quat_rotate(q, off)

    @staticmethod
    def _coord_q_to_target_pos_dof(model: newton.Model, joint_q_coords: np.ndarray) -> np.ndarray:
        """把 joint_q（coord-space）转换成 joint_target_pos（dof-space）。

        只处理“坐标维度 == dof 维度”的关节（例如常见铰链/球铰展开形式）；
        free joint（7/6）等不满足该条件的关节会被跳过。

        Args:
            model: Newton 模型。
            joint_q_coords: coord-space 的关节配置向量。

        Returns:
            np.ndarray: dof-space 的目标位置向量。
        """
        q = np.asarray(joint_q_coords, dtype=np.float64)
        dof_count = int(model.joint_dof_count)
        target = np.zeros(dof_count, dtype=np.float64)

        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        joint_count = int(qd_start.shape[0] - 1)
        for j in range(joint_count):
            q0, q1 = int(q_start[j]), int(q_start[j + 1])
            d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
            if d1 <= d0:
                continue
            if (q1 - q0) != (d1 - d0):
                continue
            target[d0:d1] = q[q0:q1]

        return target

    def render(self):
        """渲染当前帧，并（可选）绘制粒子与足端轨迹折线。"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # 粒子地形
        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.mpm_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.solver_mpm.collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses * 100.0,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        if self.show_foot_traj:
            self._log_polyline("/foot_traj/LF", self._traj_lf, wp.vec3(1.0, 0.2, 0.2))
            self._log_polyline("/foot_traj/RF", self._traj_rf, wp.vec3(0.2, 0.4, 1.0))
        else:
            self.viewer.log_lines("/foot_traj/LF", None, None, None)
            self.viewer.log_lines("/foot_traj/RF", None, None, None)

        self.viewer.end_frame()

        # 实时输出与力曲线刷新
        if self.time_history:
            fx, fy, fz = self.foot_particle_force_world
            fn = self.fn_history[-1] if self.fn_history else float(max(0.0, float(fz)))
            ft = self.ft_history[-1] if self.ft_history else float(np.sqrt(float(fx * fx + fy * fy)))
            print(
                f"\rLF: Fn={fn:8.2f} N, Ft={ft:8.2f} N (fx={fx:7.2f}, fy={fy:7.2f}, fz={fz:7.2f})",
                end="",
            )

        if (
            self.enable_plot
            and self.ax is not None
            and self.fig is not None
            and self.line_fn is not None
            and self.line_ft is not None
            and self.line_fn_filtered is not None
            and self.line_ft_filtered is not None
        ):
            self.line_fn.set_xdata(self.time_history)
            self.line_fn.set_ydata(self.fn_raw_history)
            self.line_ft.set_xdata(self.time_history)
            self.line_ft.set_ydata(self.ft_raw_history)
            self.line_fn_filtered.set_xdata(self.time_history)
            self.line_fn_filtered.set_ydata(self.fn_history)
            self.line_ft_filtered.set_xdata(self.time_history)
            self.line_ft_filtered.set_ydata(self.ft_history)
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


    def collect_collider_impulses(self) -> None:
        """从 MPM 求解器收集本帧的碰撞冲量（用于下一帧刚体外力估计）。"""
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.solver_mpm.collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self._n_colliders = int(n_colliders)
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """在指定区域发射粒子并按噪声分类为 sand/snow/mud/block。"""
        all_ids = Example._spawn_particles(
            builder,
            args,
            particle_lo=np.array(getattr(args, "emit_lo", [-0.5, -0.5, 0.0]), dtype=np.float64),
            particle_hi=np.array(getattr(args, "emit_hi", [0.5, 0.5, 0.15]), dtype=np.float64),
            density=1000,
            flags=newton.ParticleFlags.ACTIVE,
        )

        sand_ids, snow_ids, mud_ids, block_ids = Example._classify_by_noise(builder, all_ids)
        Example._assign_density(builder, sand_ids, 2500)
        Example._assign_density(builder, snow_ids, 300)
        Example._assign_density(builder, mud_ids, 1000)
        Example._assign_density(builder, block_ids, 2700)
        return np.array(sand_ids), np.array(snow_ids), np.array(mud_ids), np.array(block_ids)

    @staticmethod
    def _classify_by_noise(builder, ids):
        freq = 1.2
        octaves = 4
        noise_vals = []

        for particle_id in ids:
            q = builder.particle_q[particle_id]
            n = noise.pnoise3(q[0] * freq, q[1] * freq, q[2] * freq, octaves=octaves)
            n = (n + 1) * 0.5
            noise_vals.append(n)

        noise_vals = np.array(noise_vals)
        t1 = np.quantile(noise_vals, 0.25)
        t2 = np.quantile(noise_vals, 0.50)
        t3 = np.quantile(noise_vals, 0.75)

        sand, snow, mud, block = [], [], [], []
        for particle_id, n in zip(ids, noise_vals):
            if n <= t1:
                sand.append(particle_id)
            elif n <= t2:
                snow.append(particle_id)
            elif n <= t3:
                mud.append(particle_id)
            else:
                block.append(particle_id)

        return sand, snow, mud, block

    @staticmethod
    def _assign_density(builder, particle_ids, density):
        for particle_id in particle_ids:
            radius = builder.particle_radius[particle_id]
            volume = (4.0 / 3.0) * np.pi * radius**3
            builder.particle_mass[particle_id] = density * volume

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, args, particle_lo, particle_hi, density, flags):
        voxel_size = float(getattr(args, "voxel_size", 0.03))
        particles_per_cell = 3.0
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(cell_volume * density)

        begin_id = len(builder.particle_q)
        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(particle_res[0]) + 1,
            dim_y=int(particle_res[1]) + 1,
            dim_z=int(particle_res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=flags,
        )
        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=int)

    def test_final(self):
        """示例运行结束后的断言测试。"""
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )

    @staticmethod
    def _set_ke_kd_for_nonfree_joints(model: newton.Model, ke: float, kd: float) -> None:
        """为非 free joint 的 dof 设置 joint target 的 ke/kd。

        Args:
            model: Newton 模型。
            ke: 目标位置刚度。
            kd: 目标位置阻尼。
        """
        dof_count = int(model.joint_dof_count)
        ke_all = [0.0] * dof_count
        kd_all = [0.0] * dof_count

        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        joint_count = int(qd_start.shape[0] - 1)
        for j in range(joint_count):
            q0, q1 = int(q_start[j]), int(q_start[j + 1])
            d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
            if d1 <= d0:
                continue
            if (q1 - q0) != (d1 - d0):
                continue
            for d in range(d0, d1):
                if 0 <= d < dof_count:
                    ke_all[d] = float(ke)
                    kd_all[d] = float(kd)

        model.joint_target_ke.assign(ke_all)
        model.joint_target_kd.assign(kd_all)

    @classmethod
    def _build_default_target_pos_dof(cls, model: newton.Model, state: newton.State) -> np.ndarray:
        """从初始状态构造默认的 dof-space 目标位置向量。

        Args:
            model: Newton 模型。
            state: 初始状态（包含 joint_q）。

        Returns:
            np.ndarray: dof-space 的目标位置（通常用于“站姿/零动作”基准）。
        """
        q = state.joint_q.numpy().astype(np.float64)
        dof_count = int(model.joint_dof_count)
        target = np.zeros(dof_count, dtype=np.float64)

        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        joint_count = int(qd_start.shape[0] - 1)
        for j in range(joint_count):
            q0, q1 = int(q_start[j]), int(q_start[j + 1])
            d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
            if d1 <= d0:
                continue
            if (q1 - q0) != (d1 - d0):
                continue
            target[d0:d1] = q[q0:q1]

        return target

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """用四元数旋转向量（假定 q 为 [x,y,z,w]）。

        Args:
            q: 四元数，shape=(4,)。
            v: 向量，shape=(3,) 或 (N,3)。

        Returns:
            np.ndarray: 旋转后的向量，shape 与 v 一致。
        """
        q = np.asarray(q, dtype=np.float64).reshape(4)
        v = np.asarray(v, dtype=np.float64)
        q_xyz = q[..., :3]
        w = q[..., 3:4]
        t = 2.0 * np.cross(q_xyz, v)
        return v + w * t + np.cross(q_xyz, t)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        """返回四元数共轭（q 的逆在单位四元数情况下等于共轭）。"""
        q = np.asarray(q, dtype=np.float64).reshape(4)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def _local_to_world(self, local_pos: np.ndarray) -> np.ndarray:
        """把机器人基座局部坐标转换到世界坐标。"""
        base_p, base_q = self._get_base_pose_from_state(self.state_0)
        rotated = self._quat_rotate(base_q, np.asarray(local_pos, dtype=np.float64))
        return rotated + base_p

    def _world_to_local(self, world_pos: np.ndarray) -> np.ndarray:
        """把世界坐标转换到机器人基座局部坐标。"""
        base_p, base_q = self._get_base_pose_from_state(self.state_0)
        local = np.asarray(world_pos, dtype=np.float64) - base_p
        return self._quat_rotate(self._quat_conj(base_q), local)

    def _validate_gait_params(self) -> None:
        """检查步态参数合法性；不合法则抛异常。"""
        if self.gait_period <= 0.0:
            raise ValueError("gait_period 必须大于 0")
        if not (0.0 < self.swing_ratio < 1.0):
            raise ValueError("swing_ratio 必须在 (0, 1) 之间")
        if not (0.0 <= self.swing_apex_x1 <= 1.0 and 0.0 <= self.swing_apex_x2 <= 1.0):
            raise ValueError("swing_apex_x1 和 swing_apex_x2 必须在 [0, 1] 之间")

    @classmethod
    def _transform_points(cls, xform: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """对点集施加刚体变换（平移 + 四元数旋转）。

        Args:
            xform: 刚体变换，shape=(7,)，格式为 [px,py,pz,qx,qy,qz,qw]。
            pts: 点或点集，shape=(3,) 或 (N,3)。

        Returns:
            np.ndarray: 变换后的点，shape 与 pts 一致。
        """
        x = np.asarray(xform, dtype=np.float64).reshape(7)
        p = x[:3]
        q = x[3:]
        return cls._quat_rotate(q, pts) + p


    def _shape_vertices_local(self, model: newton.Model, shape_id: int) -> np.ndarray:
        """提取指定 shape 的局部顶点（用于估算模型最低点）。

        Args:
            model: Newton 模型。
            shape_id: shape 索引。

        Returns:
            np.ndarray: 局部顶点坐标数组，shape=(N,3)。对平面等无顶点几何返回空数组。
        """
        geo_type = int(model.shape_type.numpy()[shape_id])
        scale = np.array(model.shape_scale.numpy()[shape_id], dtype=np.float64)

        if geo_type == int(newton.GeoType.MESH) or geo_type == int(newton.GeoType.CONVEX_MESH):
            src_mesh = model.shape_source[shape_id]
            v = np.asarray(src_mesh.vertices, dtype=np.float64) * float(np.max(scale))
        elif geo_type == int(newton.GeoType.SPHERE):
            v, _i = newton.utils.create_sphere_mesh(float(scale[0]))
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.CAPSULE):
            v, _i = newton.utils.create_capsule_mesh(float(scale[0]), float(scale[1]), up_axis=2)
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.CYLINDER):
            v, _i = newton.utils.create_cylinder_mesh(float(scale[0]), float(scale[1]), up_axis=2)
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.CONE):
            v, _i = newton.utils.create_cone_mesh(float(scale[0]), float(scale[1]), up_axis=2)
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.BOX):
            v, _i = newton.utils.create_box_mesh(extents=(float(scale[0]), float(scale[1]), float(scale[2])))
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.ELLIPSOID):
            v, _i = newton.utils.create_ellipsoid_mesh(float(scale[0]), float(scale[1]), float(scale[2]))
            v = np.asarray(v, dtype=np.float64)
        elif geo_type == int(newton.GeoType.PLANE):
            return np.zeros((0, 3), dtype=np.float64)
        else:
            return np.zeros((0, 3), dtype=np.float64)

        if v.shape[0] > 8000:
            v = v[:: int(np.ceil(v.shape[0] / 8000.0))]
        return v

    def _model_min_z(self, model: newton.Model, state: newton.State) -> float:
        """估算当前状态下模型几何的世界系最低 z 值。"""
        shape_flags = model.shape_flags.numpy() if hasattr(model, "shape_flags") else None
        shape_xforms = model.shape_transform.numpy() if hasattr(model, "shape_transform") else None
        body_xforms = state.body_q.numpy()

        min_z = np.inf
        for body_id in range(int(model.body_count)):
            if body_id not in model.body_shapes:
                continue
            body_xf = body_xforms[body_id]
            for sid in model.body_shapes[body_id]:
                sid = int(sid)
                if shape_flags is not None:
                    if (int(shape_flags[sid]) & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0:
                        continue
                v_local = self._shape_vertices_local(model, sid)
                if v_local.shape[0] == 0:
                    continue
                if shape_xforms is None:
                    continue
                v_body = self._transform_points(shape_xforms[sid], v_local)
                v_world = self._transform_points(body_xf, v_body)
                min_z = min(min_z, float(np.min(v_world[:, 2])))

        if not np.isfinite(min_z):
            return float(body_xforms[0][2])
        return float(min_z)

    def _drop_to_ground(self, model: newton.Model, state: newton.State, ground_z: float, clearance: float) -> None:
        """将浮动基座沿 z 方向平移，使模型最低点落到离地 `clearance` 的高度。

        仅在模型存在 free joint 时生效。

        Args:
            model: Newton 模型。
            state: 要修改的状态（会原地更新其 joint_q）。
            ground_z: 地面高度（世界系）。
            clearance: 期望的离地最小间隙。
        """
        q_start = model.joint_q_start.numpy().astype(np.int32)
        qd_start = model.joint_qd_start.numpy().astype(np.int32)
        joint_count = int(qd_start.shape[0] - 1)

        free_joint_q0 = None
        for j in range(joint_count):
            q0, q1 = int(q_start[j]), int(q_start[j + 1])
            d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
            if (q1 - q0) == 7 and (d1 - d0) == 6:
                free_joint_q0 = q0
                break
        if free_joint_q0 is None:
            return

        min_z = self._model_min_z(model, state)
        dz = (float(ground_z) + float(clearance)) - float(min_z)
        if abs(dz) < 1e-6:
            return
        q = state.joint_q.numpy().astype(np.float64)
        q[int(free_joint_q0) + 2] += float(dz)
        state.joint_q.assign(q.tolist())

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    import argparse  # noqa: PLC0415

    parser.add_argument("--fps", type=float, default=60.0, help="渲染帧率（Hz）")
    parser.add_argument("--substeps", type=int, default=3, help="每帧的物理子步数（越大越稳定但更慢）")
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -10.0], help="重力向量")

    # 粒子地形（MPM）参数
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0], help="粒子生成区域下界")
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.15], help="粒子生成区域上界")
    parser.add_argument("--density", type=float, default=2500.0, help="基础密度（emit_particles 会按类型覆盖）")
    parser.add_argument(
        "--air-drag",
        type=float,
        default=1.0,
        help="MPM 速度阻尼（非 0 可显著缓解长程能量积累导致的 NaN/粒子突然消失）",
    )
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0)
    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e15)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.48, help="摩擦系数（同时用于刚体形状与粒子）")
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e5)
    parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--hardening", type=float, default=0.0)

    parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
    parser.add_argument("--solver", "-s", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"])
    parser.add_argument("--transfer-scheme", "-ts", type=str, default="pic", choices=["apic", "pic"])
    parser.add_argument("--strain-basis", "-sb", type=str, default="P0", choices=["P0", "Q1"])
    parser.add_argument("--max-iterations", "-it", type=int, default=50)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument(
        "--mpm-substeps",
        type=int,
        default=4,
        help="每帧 MPM 子步数（>1 可显著缓解粒子数值发散/突然消失，但更慢）",
    )
    parser.add_argument("--show-impulses", action=argparse.BooleanOptionalAction, default=False, help="显示碰撞冲量 debug 线")
    parser.add_argument(
        "--debug-mpm-health",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="打印 MPM 健康检查（NaN/Inf、active 粒子数、AABB、max|v|），用于定位粒子突然消失",
    )
    parser.add_argument(
        "--debug-mpm-health-every",
        type=int,
        default=30,
        help="MPM 健康检查打印频率（帧）",
    )
    parser.add_argument(
        "--debug-mpm-health-sample",
        type=int,
        default=0,
        help="MPM 健康检查随机采样粒子数（0=全量；采样更快但统计近似）",
    )
    parser.add_argument(
        "--debug-mpm-health-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="一旦检测到 NaN/Inf 或爆炸阈值，立刻退出（便于二分定位帧）",
    )
    parser.add_argument("--joint-target-ke", type=float, default=2000.0, help="关节目标位置控制刚度（非 free joint）")
    parser.add_argument("--v-fall", type=float, default=0.05, help="状态机 FALL 阶段下落速度（m/s）")
    parser.add_argument(
        "--explore-speed",
        type=float,
        default=1.0,
        help="状态机探索速度倍率（缩放 v-fall / v-slide；>1 更快，<1 更慢）",
    )
    parser.add_argument(
        "--sm-max-penetration",
        type=float,
        default=0.15,
        help="状态机允许足端相对 terrain_z_top 的最大下压深度（m），用于防止数值发散",
    )
    parser.add_argument("--joint-target-kd", type=float, default=50.0, help="关节目标位置控制阻尼（非 free joint）")
    parser.add_argument("--ik-iters", type=int, default=50, help="每个物理子步里 IK 的迭代次数")

    parser.add_argument(
        "--foot-link",
        type=str,
        default="LF_SHANK",
        help="Body(link) name used as the foot for force readout / particle collisions",
    )

    parser.add_argument(
        "--show-foot-traj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否显示 LF/RF 足端轨迹（debug 折线）",
    )

    parser.add_argument(
        "--control-mode",
        type=str,
        default="state-machine",
        choices=["teleop", "state-machine"],
        help="控制模式：teleop 或 state-machine",
    )

    parser.add_argument(
        "--state-machine-drive",
        type=str,
        default="ik-position",
        choices=["ik-position", "ik-velocity"],
        help="state-machine 驱动：ik-position 只写 joint_target_pos；ik-velocity 额外写 joint_target_vel（由目标差分得到）",
    )
    parser.add_argument(
        "--sm-joint-speed-max",
        type=float,
        default=6.0,
        help="state-machine 的关节目标速度限幅（dof 单位，通常 rad/s）；仅在 --state-machine-drive ik-velocity 生效；<=0 关闭限幅",
    )

    parser.add_argument("--sm-joint-slew-rate", type=float, default=4.0, help="state-machine 的关节目标位置限速（rad/s）；用于抑制高频抖动；<=0 关闭")
    parser.add_argument("--sm-joint-pos-alpha", type=float, default=0.15, help="state-machine joint_target_pos 的一阶低通系数 [0,1]；越小越平滑")
    parser.add_argument("--sm-joint-vel-alpha", type=float, default=0.15, help="state-machine joint_target_vel 的一阶低通系数 [0,1]；仅在 ik-velocity 生效")
    parser.add_argument(
        "--rigid-solver",
        type=str,
        default="xpbd",
        choices=["mujoco", "xpbd"],
        help="刚体求解器：mujoco 或 xpbd",
    )

    # XPBD 稳定性参数
    parser.add_argument("--xpbd-iterations", type=int, default=80, help="XPBD 迭代次数（越大越稳但更慢）")
    parser.add_argument("--xpbd-rigid-contact-relaxation", type=float, default=0.3, help="XPBD 刚体接触松弛系数（更小更稳/更粘）")
    parser.add_argument("--xpbd-joint-linear-relaxation", type=float, default=0.3, help="XPBD 关节线性松弛系数")
    parser.add_argument("--xpbd-joint-angular-relaxation", type=float, default=0.3, help="XPBD 关节角向松弛系数")
    parser.add_argument("--xpbd-angular-damping", type=float, default=0.4, help="XPBD 角速度阻尼（0~1 之间通常更稳）")
    parser.add_argument("--xpbd-joint-target-ke", type=float, default=600.0, help="XPBD 下关节目标位置刚度（非 free joint）；更小更顺滑")
    parser.add_argument("--xpbd-joint-target-kd", type=float, default=80.0, help="XPBD 下关节目标位置阻尼（非 free joint）；更大更稳")
    parser.add_argument("--initial-state", type=int, default=STATE_STEP, help="状态机初始状态")
    parser.add_argument("--fz-contact-threshold", type=float, default=5, help="触地力阈值 (N)")
    parser.add_argument("--fz-filter-alpha", type=float, default=0.2, help="Fz 一阶滤波系数 [0,1]")
    parser.add_argument("--force-max", type=float, default=200.0, help="单腿力限幅 (N)")
    parser.add_argument("--touch-count-th", type=int, default=2, help="触地判定连续帧阈值")
    parser.add_argument("--fz-ref-slide", type=float, default=10.0, help="滑动阶段法向力参考 (N)")
    parser.add_argument("--kp-z", type=float, default=800.0, help="滑动阶段法向力 P 控制系数")
    parser.add_argument("--v-slide", type=float, default=0.01, help="滑动阶段水平速度 (m/s)")

    parser.add_argument("--tri-step-len", type=float, default=0.15, help="三角步态最大步长 (m)")
    parser.add_argument("--tri-step-h", type=float, default=0.05, help="三角步态抬腿高度 (m)")
    parser.add_argument("--candidate-num", type=int, default=4, help="候选落足点数量")
    parser.add_argument(
        "--candidate-param",
        type=str,
        default="hip-angle",
        choices=["hip-angle", "xy-distance"],
        help="候选落足点取样参数：hip-angle=按 LF_HAA 累计转角等分；xy-distance=按索引等分并按 XY 距离从远到近",
    )
    parser.add_argument("--stability-check-max-frames", type=int, default=10, help="稳定性检测总帧数")
    parser.add_argument("--stability-check-th", type=int, default=7, help="稳定性判定通过所需连续帧数")
    parser.add_argument("--x-slide-th", type=float, default=0.002, help="滑动阈值 (m)")
    parser.add_argument("--z-sink-th", type=float, default=0.002, help="沉陷阈值 (m)")
    parser.add_argument("--check-press-depth", type=float, default=0.05, help="状态2检测时的下压深度 (m)")
    parser.add_argument("--check-press-speed", type=float, default=0.2, help="状态2下压速度 (m/s)")
    parser.add_argument("--check-retract-speed", type=float, default=0.1, help="状态2回拉速度 (m/s)")
    parser.add_argument(
        "--check-outward-y",
        type=float,
        default=0.1,
        help="状态2下压/保持时的横向外扩偏置（m）；用于抑制内收并保持足端 XY 基本不变",
    )

    parser.add_argument("--sm-explore-distance", type=float, default=0.2, help="扇形扫描：单次探索最大半径 (m)")
    parser.add_argument("--sm-explore-distance-step", type=float, default=0.1, help="扇形扫描：扫完整扇形后半径扩张步长 (m)")
    parser.add_argument("--sm-explore-start-angle-deg", type=float, default=-90.0, help="扇形扫描：起始角度（度）")
    parser.add_argument("--sm-explore-end-angle-deg", type=float, default=90.0, help="扇形扫描：终止角度（度）")
    parser.add_argument("--sm-explore-angle-step-deg", type=float, default=15.0, help="扇形扫描：角度步进（度）")
    parser.add_argument("--sm-lift-height", type=float, default=0.05, help="探索失败后抬足高度 (m)")

    parser.add_argument(
        "--enable-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否实时绘制 LF 交互力-时间曲线（matplotlib，可选）",
    )

    parser.add_argument("--settle-time", type=float, default=0.5, help="开始行走前原地站稳的时间（秒），期间锁足端初始落点")

    parser.add_argument("--step-length", type=float, default=0.4, help="步长（髋局部 x 方向前后摆动幅度）")
    parser.add_argument("--step-height", type=float, default=0.1, help="抬脚高度（髋局部 z 方向）")
    parser.add_argument("--swing-apex-x1", type=float, default=0.25, help="Bezier 控制点 p1 的 x 位置比例 [0,1]（越小越早抬脚）")
    parser.add_argument("--swing-apex-x2", type=float, default=0.75, help="Bezier 控制点 p2 的 x 位置比例 [0,1]（越大越晚落脚）")

    # Bezier 横向（y）偏移：这里的 y 是“向外侧”的幅度，会自动对左右腿做镜像：左腿乘 +1，右腿乘 -1。
    # 你想调 p0~p3 的 y，就改下面四个值即可。
    parser.add_argument("--swing-y0", type=float, default=0.0, help="Bezier 控制点 p0 的 y 偏移幅度（自动左右镜像）")
    parser.add_argument("--swing-y1", type=float, default=0.1, help="Bezier 控制点 p1 的 y 偏移幅度（自动左右镜像）")
    parser.add_argument("--swing-y2", type=float, default=0.1, help="Bezier 控制点 p2 的 y 偏移幅度（自动左右镜像）")
    parser.add_argument("--swing-y3", type=float, default=0.0, help="Bezier 控制点 p3 的 y 偏移幅度（自动左右镜像）")
    parser.add_argument(
        "--auto-drop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在初始化时自动把机器人下放到地面上（仅在 --robot-floating=true 时生效）",
    )
    parser.add_argument("--drop-clearance", type=float, default=0.002, help="自动下放到地面后，离地的最小间隙")

    default_urdf = os.path.join(os.path.dirname(__file__), "el_mini", "urdf", "el_mini_allmove.urdf")
    parser.add_argument("--robot-urdf", type=str, default=default_urdf, help="机器人 URDF 路径")
    parser.add_argument(
        "--robot-floating",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="以浮动基座导入机器人（启用 auto-drop）",
    )
    parser.add_argument("--robot-base-z", type=float, default=0.35, help="导入时 base 的初始高度（z）")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)