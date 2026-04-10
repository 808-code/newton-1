from __future__ import annotations

# pyright: reportInvalidTypeForm=false

import os
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from collections import deque

import noise
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM


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
            xform=wp.transform(wp.vec3(0.0, 0.0 ,float(base_z)), wp.quat_identity()),
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

        self.solver_block = newton.solvers.SolverXPBD(self.model, iterations=10)

        # MPM 状态与求解器
        self.sand_state_0 = self.mpm_model.state()
        self.solver_mpm = SolverImplicitMPM(mpm_model, mpm_options)
        self.solver_mpm.enrich_state(self.sand_state_0)

        self.contacts = self.model.collide(self.state_0)

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
                weight=1.0,
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

        
        self._ik_joint_limits = newton.ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
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

        # host 侧遥操作会导致 host<->device 数据交换，不适配 CUDA Graph capture
        self._disable_cuda_graph = True

        # ------------------------------------------------------------------
        # 足端-粒子接触力统计（实时读取 + 曲线）
        # ------------------------------------------------------------------
        self.foot_particle_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.foot_particle_force_world = np.zeros(3, dtype=np.float32)
        self.fn_history: list[float] = []
        self.ft_history: list[float] = []
        self.time_history: list[float] = []
        self.enable_plot = bool(getattr(options, "enable_plot", True))
        self._plt = _try_import_matplotlib_pyplot() if self.enable_plot else None
        self.fig = None
        self.ax = None
        self.line_fn = None
        self.line_ft = None
        if self._plt is None:
            self.enable_plot = False
        if self.enable_plot:
            self._plt.ion()
            self.fig, self.ax = self._plt.subplots()
            (self.line_fn,) = self.ax.plot(self.time_history, self.fn_history, label="LF Fn (from fz)")
            (self.line_ft,) = self.ax.plot(self.time_history, self.ft_history, label="LF Ft (from fx/fy)")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Force (N)")
            self.ax.set_title("Interaction Force vs. Time")
            self.ax.grid(True)
            self.ax.legend(loc="upper right")

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

            self.contacts = self.model.collide(self.state_0)
            self.solver_block.step(self.state_0, self.state_1, control=self.control, contacts=self.contacts, dt=self.sim_dt)
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

        self.solver_mpm.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.frame_dt)
        self.collect_collider_impulses()

        self.foot_particle_force.zero_()
        if self._n_colliders > 0:
            wp.launch(
                compute_single_body_contact_force,
                dim=int(self._n_colliders),
                inputs=[
                    float(self.frame_dt),
                    int(self.foot),
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_body_id,
                    self.foot_particle_force,
                ],
            )
            self.foot_particle_force_world = self.foot_particle_force.numpy()[0].astype(np.float32)

    def step(self) -> None:
        """主仿真步骤：结构对齐 robot_terrain。"""
        if getattr(self, "graph", None):
            wp.capture_launch(self.graph)
        else:
            self.simulate_leg()

        if getattr(self, "sand_graph", None):
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_mpm()

        self.sim_time += self.frame_dt
        fx, fy, fz = self.foot_particle_force_world
        fn = float(max(0.0, float(fz)))
        ft = float(np.sqrt(float(fx * fx + fy * fy)))
        force_magnitude = float(np.sqrt(float(fx * fx + fy * fy + fz * fz)))
        self.fn_history.append(fn)
        self.ft_history.append(ft)
        self.time_history.append(float(self.sim_time))

        if self._headless_null_viewer:
            if force_magnitude > 1e-6:
                self._contact_seen_frames += 1
            else:
                self._contact_seen_frames = 0
            if self._contact_seen_frames >= 3:
                print("\nContact detected; stopping early.")
                raise SystemExit(0)


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
        ):
            self.line_fn.set_xdata(self.time_history)
            self.line_fn.set_ydata(self.fn_history)
            self.line_ft.set_xdata(self.time_history)
            self.line_ft.set_ydata(self.ft_history)
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
            particle_hi=np.array(getattr(args, "emit_hi", [0.5, 0.5, 0.25]), dtype=np.float64),
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
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.25], help="粒子生成区域上界")
    parser.add_argument("--density", type=float, default=2500.0, help="基础密度（emit_particles 会按类型覆盖）")
    parser.add_argument("--air-drag", type=float, default=0.0)
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
    parser.add_argument("--show-impulses", action=argparse.BooleanOptionalAction, default=False, help="显示碰撞冲量 debug 线")
    parser.add_argument("--joint-target-ke", type=float, default=2000.0, help="关节目标位置控制刚度（非 free joint）")
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
        default=False,
        help="是否显示 LF/RF 足端轨迹（debug 折线）",
    )

    parser.add_argument(
        "--enable-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否实时绘制 LF 交互力-时间曲线（matplotlib，可选）",
    )

    parser.add_argument("--settle-time", type=float, default=1.0, help="开始行走前原地站稳的时间（秒），期间锁足端初始落点")

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
    parser.add_argument("--robot-base-z", type=float, default=0.45, help="导入时 base 的初始高度（z）")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)