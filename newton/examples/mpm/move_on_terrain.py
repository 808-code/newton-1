from __future__ import annotations

# pyright: reportInvalidTypeForm=false

import os
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass

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
    name: str
    hip_body: int
    foot_body: int
    foot_offset_local: np.ndarray
    side_sign: float
    phase_offset: float
    target_pos: wp.array
    pos_obj: newton.ik.IKPositionObjective
    foot_rel_base0: np.ndarray | None = None


class Example:
    def __init__(self, viewer, options):
        self.fps = float(getattr(options, "fps", 50.0))
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = int(getattr(options, "substeps", 4))
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        # self._headless_null_viewer = self.viewer.__class__.__name__.lower().endswith("null")
        # self._contact_seen_frames = 0

        emit_hi = getattr(options, "emit_hi", [0.5, 0.5, 0.15])
        voxel_size = float(getattr(options, "voxel_size", 0.03))
        self.terrain_z_top = float(emit_hi[2]) + 0.5 * voxel_size

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3

        builder.default_shape_cfg.mu = 0.75
        friction_coeff = float(getattr(options, "friction_coeff", 0.75))

        ground_shape = builder.add_ground_plane()
        self.disable_ground_shape_contact = bool(getattr(options, "disable_ground_shape_contact", False))
        if self.disable_ground_shape_contact and hasattr(builder, "shape_flags"):
            builder.shape_flags[ground_shape] = builder.shape_flags[ground_shape] & ~newton.ShapeFlags.COLLIDE_SHAPES

        this_dir = os.path.dirname(__file__)
        default_urdf = os.path.join(this_dir, "el_mini", "urdf", "el_mini_allmove.urdf")
        robot_urdf = str(getattr(options, "robot_urdf", default_urdf))
        self._robot_urdf_path = robot_urdf
        robot_floating = bool(getattr(options, "robot_floating", True))
        base_z = float(getattr(options, "robot_base_z", 1.0))

        builder.add_urdf(
            robot_urdf,
            xform=wp.transform(wp.vec3(-1.2, 0.0, base_z), wp.quat_identity()),
            floating=robot_floating,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        self._builder_joint_key = [str(k) for k in list(getattr(builder, "joint_key", []))]

        self.collide_shanks = bool(getattr(options, "collide_shanks", True))
        self._filter_particle_collision_shapes(
            builder=builder,
            ground_shape=int(ground_shape),
            collide_shanks=self.collide_shanks,
        )

        mpm_builder = newton.ModelBuilder()
        sand_particles, snow_particles, mud_particles, block_particles = Example.emit_particles(mpm_builder, options)

        self.model = builder.finalize()
        self.mpm_model = mpm_builder.finalize()

        gravity = getattr(options, "gravity", [0.0, 0.0, -10.0])
        self.model.set_gravity(gravity)
        self.mpm_model.set_gravity(gravity)
        self.mpm_model.particle_mu = friction_coeff
        self.mpm_model.particle_kd = 0.0
        self.mpm_model.particle_ke = 1.0e15

        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))
        if hasattr(mpm_options, "grid_type") and (not wp.get_device().is_cuda):
            if getattr(mpm_options, "grid_type", None) == "sparse":
                mpm_options.grid_type = "fixed"

        mpm_model = SolverImplicitMPM.Model(self.mpm_model, mpm_options)

        sand_particles = wp.array(sand_particles, dtype=int, device=self.mpm_model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=self.mpm_model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=self.mpm_model.device)
        block_particles = wp.array(block_particles, dtype=int, device=self.mpm_model.device)

        mpm_model.material_parameters.yield_pressure[snow_particles].fill_(2.0e4)
        mpm_model.material_parameters.yield_stress[snow_particles].fill_(1.0e3)
        mpm_model.material_parameters.tensile_yield_ratio[snow_particles].fill_(0.05)
        mpm_model.material_parameters.friction[snow_particles].fill_(0.1)
        mpm_model.material_parameters.hardening[snow_particles].fill_(10.0)

        mpm_model.material_parameters.yield_pressure[mud_particles].fill_(1.0e10)
        mpm_model.material_parameters.yield_stress[mud_particles].fill_(3.0e2)
        mpm_model.material_parameters.tensile_yield_ratio[mud_particles].fill_(1.0)
        mpm_model.material_parameters.hardening[mud_particles].fill_(2.0)
        mpm_model.material_parameters.friction[mud_particles].fill_(0.0)

        mpm_model.material_parameters.yield_pressure[block_particles].fill_(1.0e8)
        mpm_model.material_parameters.yield_stress[block_particles].fill_(1.0e7)
        mpm_model.material_parameters.tensile_yield_ratio[block_particles].fill_(0.8)
        mpm_model.material_parameters.friction[block_particles].fill_(0.6)
        mpm_model.material_parameters.hardening[block_particles].fill_(20.0)

        mpm_model.notify_particle_material_changed()
        mpm_model.setup_collider(model=self.model)

        

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sand_state_0 = self.mpm_model.state()
        

        if self.model.joint_target_ke is not None and self.model.joint_target_kd is not None:
            ke = float(getattr(options, "joint_target_ke", 200.0))
            kd = float(getattr(options, "joint_target_kd", 100.0))
            self._set_ke_kd_for_nonfree_joints(self.model, ke=ke, kd=kd)

        self._default_target_pos_dof = self._build_default_target_pos_dof(self.model, self.state_0)
        dof_count = int(self.model.joint_dof_count)

        self._control_dtype_pos = self.control.joint_target_pos.dtype
        self._control_dtype_vel = self.control.joint_target_vel.dtype
        self._set_control_targets(
            target_pos_dof=self._default_target_pos_dof,
            target_vel_dof=np.zeros(dof_count, dtype=np.float64),
        )
        
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        auto_drop = bool(getattr(options, "auto_drop", True))
        if robot_floating and auto_drop:
            clearance = float(getattr(options, "drop_clearance", 0.002))
            self._drop_to_ground(self.model, self.state_0, ground_z=0.0, clearance=clearance)
            self.state_1.joint_q.assign(self.state_0.joint_q.numpy().tolist())
            if self.state_0.joint_qd is not None and self.state_1.joint_qd is not None:
                self.state_1.joint_qd.assign(self.state_0.joint_qd.numpy().tolist())
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.solver_block = newton.solvers.SolverMuJoCo(self.model, ls_parallel=True, njmax=50)
        self.solver_mpm = SolverImplicitMPM(mpm_model, mpm_options)

        self.solver_mpm.enrich_state(self.sand_state_0)

        self.show_foot_traj = bool(getattr(options, "show_foot_traj", False))
        self._foot_traj_maxlen = int(max(2, round(self.fps * 5.0)))
        self._traj_lf = deque(maxlen=self._foot_traj_maxlen)
        self._traj_rf = deque(maxlen=self._foot_traj_maxlen)
        
        self.settle_time = float(getattr(options, "settle_time", 0.8))
        self.gait_period = float(getattr(options, "gait_period", 1.0))
        self.swing_ratio = float(getattr(options, "swing_ratio", 0.5))
        self.step_length = float(getattr(options, "step_length", 0.15))
        self.step_height = float(getattr(options, "step_height", 0.2))
        self.swing_apex_x1 = float(getattr(options, "swing_apex_x1", 0.5))
        self.swing_apex_x2 = float(getattr(options, "swing_apex_x2", 0.7))
        self.swing_y0 = float(getattr(options, "swing_y0", 0.0))
        self.swing_y1 = float(getattr(options, "swing_y1", 0.1))
        self.swing_y2 = float(getattr(options, "swing_y2", 0.1))
        self.swing_y3 = float(getattr(options, "swing_y3", 0.0))
        self._validate_gait_params()

        device = self.model.device

        self._free_joint_q0 = self._find_free_joint_q0(self.model)

        tripod_a = {"LF", "RM", "LB"}
        tripod_b = {"RF", "LM", "RB"}
        leg_order = ["RF", "RM", "RB", "LF", "LM", "LB"]

        self.legs: list[_LegSpec] = []
        pos_objs: list[newton.ik.IKPositionObjective] = []
        device = self.model.device

        for leg in leg_order:
            if leg in tripod_a:
                phase_offset = 0.0
            elif leg in tripod_b:
                phase_offset = 0.5
            else:
                phase_offset = 0.0

            hip_body = self._find_body_index(self.model, f"{leg}_HIP")
            foot_body = self._find_body_index_optional(self.model, f"{leg}_FOOT")

            foot_offset_local = np.zeros(3, dtype=np.float64)
            if foot_body is None:
                foot_body = self._find_body_index(self.model, f"{leg}_SHANK")
                foot_offset_local = self._urdf_joint_origin_xyz(
                    self._robot_urdf_path,
                    parent_link=f"{leg}_SHANK",
                    child_link=f"{leg}_FOOT",
                )

            side_sign = 1.0 if leg.startswith("L") else -1.0

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
                    side_sign=float(side_sign),
                    phase_offset=float(phase_offset),
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
        self._ik_solution_q = wp.zeros((1, int(self.model.joint_coord_count)), dtype=wp.float32, device=self.model.device)
        self._ik_iters = int(getattr(options, "ik_iters", 50))

        # self._setup_ik_controller(options)

        max_nodes = 1 << 20
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self._n_colliders = 0
        self.collect_collider_impulses()

        self.collider_body_id = mpm_model.collider.collider_body_index
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        lf_leg = next((l for l in self.legs if l.name == "LF"), None)
        self.foot = int(lf_leg.foot_body) if lf_leg is not None else 0
        rf_leg = next((l for l in self.legs if l.name == "RF"), None)
        self.foot_rf = int(rf_leg.foot_body) if rf_leg is not None else None
        self.foot_particle_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.foot_particle_force_world = np.zeros(3, dtype=np.float32)
        self.rf_particle_force = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.rf_particle_force_world = np.zeros(3, dtype=np.float32)

        self.particle_render_colors = wp.full(
            self.mpm_model.particle_count,
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3,
            device=self.mpm_model.device,
        )
        self.particle_render_colors[sand_particles].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_render_colors[snow_particles].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_render_colors[mud_particles].fill_(wp.vec3(0.4, 0.25, 0.25))
        self.particle_render_colors[block_particles].fill_(wp.vec3(0.5, 0.5, 0.5))

        self.force_history: list[float] = []
        self.rf_force_history: list[float] = []
        self.time_history: list[float] = []
        self.show_impulses = bool(getattr(options, "show_impulses", False))

        self.enable_plot = bool(getattr(options, "enable_plot", True))
        self.plot_lf_force = bool(getattr(options, "plot_lf_force", True))
        self.plot_rf_force = bool(getattr(options, "plot_rf_force", True))
        self._plt = None
        self.fig = None
        self.ax = None
        self.line_lf = None
        self.line_rf = None
        if self.enable_plot and (self.plot_lf_force or self.plot_rf_force):
            self._plt = _try_import_matplotlib_pyplot()
            if self._plt is not None:
                self._plt.ion()
                self.fig, self.ax = self._plt.subplots()
                if self.plot_lf_force:
                    (self.line_lf,) = self.ax.plot(self.time_history, self.force_history, label="LF")
                if self.plot_rf_force:
                    (self.line_rf,) = self.ax.plot(self.time_history, self.rf_force_history, label="RF")
                self.ax.set_xlabel("Time (s)")
                self.ax.set_ylabel("Force (N)")
                self.ax.set_title("Interaction Force vs. Time")
                self.ax.grid(True)
                if self.plot_lf_force and self.plot_rf_force:
                    self.ax.legend()
            else:
                self.enable_plot = False

        self.graph = None
        self.sand_graph = None

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    def _setup_ik_controller(self, options) -> None:
        self._free_joint_q0 = self._find_free_joint_q0(self.model)

        tripod_a = {"LF", "RM", "LB"}
        tripod_b = {"RF", "LM", "RB"}
        leg_order = ["RF", "RM", "RB", "LF", "LM", "LB"]

        self.legs: list[_LegSpec] = []
        pos_objs: list[newton.ik.IKPositionObjective] = []
        device = self.model.device

        for leg in leg_order:
            if leg in tripod_a:
                phase_offset = 0.0
            elif leg in tripod_b:
                phase_offset = 0.5
            else:
                phase_offset = 0.0

            hip_body = self._find_body_index(self.model, f"{leg}_HIP")
            foot_body = self._find_body_index_optional(self.model, f"{leg}_FOOT")

            foot_offset_local = np.zeros(3, dtype=np.float64)
            if foot_body is None:
                foot_body = self._find_body_index(self.model, f"{leg}_SHANK")
                foot_offset_local = self._urdf_joint_origin_xyz(
                    self._robot_urdf_path,
                    parent_link=f"{leg}_SHANK",
                    child_link=f"{leg}_FOOT",
                )

            side_sign = 1.0 if leg.startswith("L") else -1.0

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
                    side_sign=float(side_sign),
                    phase_offset=float(phase_offset),
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
        self._ik_solution_q = wp.zeros((1, int(self.model.joint_coord_count)), dtype=wp.float32, device=self.model.device)
        self._ik_iters = int(getattr(options, "ik_iters", 50))

    def _filter_particle_collision_shapes(
        self,
        builder: newton.ModelBuilder,
        ground_shape: int,
        *,
        collide_shanks: bool,
    ) -> None:
        if not (hasattr(builder, "body_key") and hasattr(builder, "body_shapes") and hasattr(builder, "shape_flags")):
            return

        body_names = [str(k) for k in list(builder.body_key)]
        leg_prefixes = ("LF", "RF", "LM", "RM", "LB", "RB")

        def _pick_body(prefix: str, tokens: tuple[str, ...]) -> int | None:
            prefix_u = prefix.upper()
            for name in body_names:
                name_u = name.upper()
                if name_u.startswith(prefix_u) and any(token in name_u for token in tokens):
                    return int(builder.body_key.index(name))
            return None

        foot_bodies: list[int] = []
        for prefix in leg_prefixes:
            idx = _pick_body(prefix, ("FOOT", "TOE", "SOLE"))
            if idx is None:
                idx = _pick_body(prefix, ("SHANK",))
            if idx is not None:
                foot_bodies.append(idx)

        shank_bodies: list[int] = []
        if collide_shanks:
            for prefix in leg_prefixes:
                idx = _pick_body(prefix, ("SHANK",))
                if idx is not None:
                    shank_bodies.append(idx)

        if not foot_bodies:
            for i, name in enumerate(body_names):
                name_u = name.upper()
                if ("FOOT" in name_u) or ("TOE" in name_u) or ("SOLE" in name_u):
                    foot_bodies.append(i)

        foot_bodies = sorted(set(foot_bodies))
        shank_bodies = sorted(set(shank_bodies))

        keep_shape_ids: list[int] = []
        if hasattr(builder, "shape_key"):
            for body in foot_bodies:
                for shape_id in builder.body_shapes[body]:
                    key = str(builder.shape_key[shape_id]).upper()
                    if ("FOOT" in key) or ("TOE" in key) or ("SOLE" in key):
                        keep_shape_ids.append(int(shape_id))

        if collide_shanks:
            for body in shank_bodies:
                keep_shape_ids.extend(int(s) for s in builder.body_shapes[body])

        if not keep_shape_ids:
            for body in foot_bodies:
                keep_shape_ids.extend(int(s) for s in builder.body_shapes[body])
            if collide_shanks:
                for body in shank_bodies:
                    keep_shape_ids.extend(int(s) for s in builder.body_shapes[body])

        keep_shape_set = set(keep_shape_ids)
        keep_shape_set.add(int(ground_shape))
        for body in range(builder.body_count):
            for shape in builder.body_shapes[body]:
                if int(shape) not in keep_shape_set:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

    def collect_collider_impulses(self) -> None:
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
    def _cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float) -> np.ndarray:
        u = float(np.clip(u, 0.0, 1.0))
        b0 = (1.0 - u) ** 3
        b1 = 3.0 * (1.0 - u) ** 2 * u
        b2 = 3.0 * (1.0 - u) * u**2
        b3 = u**3
        return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3

    def _phase01(self, t: float, phase_offset: float) -> float:
        return float((t / self.gait_period + phase_offset) % 1.0)

    def _swing_delta_base(self, u01: float, side_sign: float) -> np.ndarray:
        
        u01 = float(np.clip(u01, 0.0, 1.0))
        x0 = -0.5 * self.step_length
        x3 = 0.5 * self.step_length

        s = float(1.0 if side_sign >= 0.0 else -1.0)

        p0 = np.array([x0, 0.0, 0.0], dtype=np.float64)
        p0[1] = s * self.swing_y0
        p1 = np.array(
            [x0 + (x3 - x0) * self.swing_apex_x1, s * self.swing_y1, self.step_height],
            dtype=np.float64,
        )
        p2 = np.array(
            [x0 + (x3 - x0) * self.swing_apex_x2, s * self.swing_y2, self.step_height],
            dtype=np.float64,
        )
        p3 = np.array([x3, s * self.swing_y3, 0.0], dtype=np.float64)

        return self._cubic_bezier(p0, p1, p2, p3, u01)

    def _stance_delta_base(self, u01: float) -> np.ndarray:
        u01 = float(np.clip(u01, 0.0, 1.0))
        x3 = 0.5 * self.step_length
        x0 = -0.5 * self.step_length
        x = (1.0 - u01) * x3 + u01 * x0
        return np.array([x, 0.0, 0.0], dtype=np.float64)

    def _set_control_targets(self, *, target_pos_dof: np.ndarray, target_vel_dof: np.ndarray | None) -> None:
        target_pos = np.asarray(target_pos_dof, dtype=np.float64)
        pos_wp = wp.array(target_pos.astype(np.float32), dtype=self._control_dtype_pos, device=self.model.device)
        wp.copy(self.control.joint_target_pos, pos_wp)

        if target_vel_dof is not None:
            target_vel = np.asarray(target_vel_dof, dtype=np.float64)
            vel_wp = wp.array(target_vel.astype(np.float32), dtype=self._control_dtype_vel, device=self.model.device)
            wp.copy(self.control.joint_target_vel, vel_wp)

    def _solve_ik_and_write_targets(self) -> None:
        seed_q = wp.array(self.state_0.joint_q, shape=(1, int(self.model.joint_coord_count)))
        self._ik_solver.reset()
        self._ik_solver.step(seed_q, self._ik_solution_q, iterations=self._ik_iters)

        q_sol = np.asarray(self._ik_solution_q.numpy()[0], dtype=np.float64)
        target_pos_dof = self._coord_q_to_target_pos_dof(self.model, q_sol)
        self._set_control_targets(target_pos_dof=target_pos_dof, target_vel_dof=None)

    def apply_control_stand(self) -> None:
        body_q_np = self.state_0.body_q.numpy()
        base_p, base_q = self._get_base_pose_from_state(self.state_0)

        for leg in self.legs:
            foot_p = self._foot_point_world(body_q_np, leg)

            if leg.foot_rel_base0 is None:
                leg.foot_rel_base0 = self._quat_rotate(self._quat_conj(base_q), (foot_p - base_p))

            target_world = base_p + self._quat_rotate(base_q, leg.foot_rel_base0)
            leg.pos_obj.set_target_position(
                0,
                wp.vec3(float(target_world[0]), float(target_world[1]), float(target_world[2])),
            )

        self._solve_ik_and_write_targets()

    def apply_control_walk(self) -> None:
        body_q_np = self.state_0.body_q.numpy()
        base_p, base_q = self._get_base_pose_from_state(self.state_0)

        for leg in self.legs:
            foot_p = self._foot_point_world(body_q_np, leg)

            if leg.foot_rel_base0 is None:
                leg.foot_rel_base0 = self._quat_rotate(self._quat_conj(base_q), (foot_p - base_p))

            ph = self._phase01(self.sim_time, leg.phase_offset)
            if ph < self.swing_ratio:
                u = ph / self.swing_ratio
                delta_base = self._swing_delta_base(u, leg.side_sign)
            else:
                u = (ph - self.swing_ratio) / (1.0 - self.swing_ratio)
                delta_base = self._stance_delta_base(u)

            target_world = base_p + self._quat_rotate(base_q, (leg.foot_rel_base0 + delta_base))
            leg.pos_obj.set_target_position(
                0,
                wp.vec3(float(target_world[0]), float(target_world[1]), float(target_world[2])),
            )

        self._solve_ik_and_write_targets()

    def simulate_leg(self) -> None:
        if self.sim_time < self.settle_time:
            self.apply_control_stand()
        else:
            self.apply_control_walk()

        self.body_sand_forces.zero_()
        if self._n_colliders > 0:
            wp.launch(
                compute_body_forces,
                dim=self._n_colliders,
                inputs=[
                    self.frame_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.body_sand_forces,
                ],
            )

        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            # wp.copy(self.state_0.body_f, self.body_sand_forces)
            self.viewer.apply_forces(self.state_0)

            self.solver_block.step(self.state_0, self.state_1, control=self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_mpm(self) -> None:
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

        self.rf_particle_force.zero_()
        if self._n_colliders > 0 and self.foot_rf is not None:
            wp.launch(
                compute_single_body_contact_force,
                dim=self._n_colliders,
                inputs=[
                    self.frame_dt,
                    int(self.foot_rf),
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_body_id,
                    self.rf_particle_force,
                ],
            )
            self.rf_particle_force_world = self.rf_particle_force.numpy()[0].astype(np.float32)

    def _record_lf_rf_foot_positions(self) -> None:
        body_q_np = self.state_0.body_q.numpy()
        lf = next((l for l in self.legs if l.name == "LF"), None)
        rf = next((l for l in self.legs if l.name == "RF"), None)
        if lf is not None:
            self._traj_lf.append(self._foot_point_world(body_q_np, lf))
        if rf is not None:
            self._traj_rf.append(self._foot_point_world(body_q_np, rf))

    def step(self) -> None:
        self.simulate_leg()
        self.simulate_mpm()
        if self.show_foot_traj:
            self._record_lf_rf_foot_positions()

        self.sim_time += self.frame_dt
        force_magnitude = float(np.linalg.norm(self.foot_particle_force_world))
        rf_force_magnitude = float(np.linalg.norm(self.rf_particle_force_world))
        self.force_history.append(force_magnitude)
        self.rf_force_history.append(rf_force_magnitude)
        self.time_history.append(self.sim_time)

        # if self._headless_null_viewer:
        #     if force_magnitude > 1e-6:
        #         self._contact_seen_frames += 1
        #     else:
        #         self._contact_seen_frames = 0
        #     if self._contact_seen_frames >= 3:
        #         raise SystemExit(0)

    def _log_polyline(self, path: str, pts, color) -> None:
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
  
    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
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

        if self.force_history:
            parts: list[str] = []
            if self.plot_lf_force:
                current_force = self.force_history[-1]
                fx, fy, fz = self.foot_particle_force_world
                parts.append(
                    f"LF: {current_force:8.2f} N (fx={fx:7.2f}, fy={fy:7.2f}, fz={fz:7.2f})"
                )
            if self.plot_rf_force:
                current_force = self.rf_force_history[-1] if self.rf_force_history else 0.0
                fx, fy, fz = self.rf_particle_force_world
                parts.append(
                    f"RF: {current_force:8.2f} N (fx={fx:7.2f}, fy={fy:7.2f}, fz={fz:7.2f})"
                )
            if parts:
                print("\r" + " | ".join(parts), end="")

            if self.enable_plot and self.ax is not None and self.fig is not None:
                if self.line_lf is not None:
                    self.line_lf.set_xdata(self.time_history)
                    self.line_lf.set_ydata(self.force_history)
                if self.line_rf is not None:
                    self.line_rf.set_xdata(self.time_history)
                    self.line_rf.set_ydata(self.rf_force_history)
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def render_ui(self, imgui) -> None:
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    def test_final(self) -> None:
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the deep terrain floor",
            lambda q, qd: q[2] > -0.2,
        )

    @staticmethod
    def _find_free_joint_q0(model: newton.Model) -> int | None:
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
        if self._free_joint_q0 is None:
            return np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        q = state.joint_q.numpy().astype(np.float64)
        q0 = int(self._free_joint_q0)
        base_p = q[q0 : q0 + 3]
        base_q = q[q0 + 3 : q0 + 7]
        return base_p, base_q

    @staticmethod
    def _find_body_index(model: newton.Model, body_name: str) -> int:
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
        if not hasattr(model, "body_key"):
            return None
        keys = [str(k) for k in list(model.body_key)]
        try:
            return int(keys.index(str(body_name)))
        except ValueError:
            return None

    @staticmethod
    def _urdf_joint_origin_xyz(urdf_path: str, *, parent_link: str, child_link: str) -> np.ndarray:
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

    @staticmethod
    def _set_ke_kd_for_nonfree_joints(model: newton.Model, ke: float, kd: float) -> None:
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
        q = np.asarray(q, dtype=np.float64).reshape(4)
        v = np.asarray(v, dtype=np.float64)
        q_xyz = q[..., :3]
        w = q[..., 3:4]
        t = 2.0 * np.cross(q_xyz, v)
        return v + w * t + np.cross(q_xyz, t)

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)

    def _validate_gait_params(self) -> None:
        if self.gait_period <= 0.0:
            raise ValueError("gait_period must be > 0")
        if not (0.0 < self.swing_ratio < 1.0):
            raise ValueError("swing_ratio must be in (0,1)")
        if not (0.0 <= self.swing_apex_x1 <= 1.0 and 0.0 <= self.swing_apex_x2 <= 1.0):
            raise ValueError("swing_apex_x1 and swing_apex_x2 must be in [0,1]")

    @classmethod
    def _transform_points(cls, xform: np.ndarray, pts: np.ndarray) -> np.ndarray:
        x = np.asarray(xform, dtype=np.float64).reshape(7)
        p = x[:3]
        q = x[3:]
        return cls._quat_rotate(q, pts) + p

    def _shape_vertices_local(self, model: newton.Model, shape_id: int) -> np.ndarray:
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

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        all_ids = Example._spawn_particles(
            builder,
            args,
            particle_lo=np.array(getattr(args, "emit_lo", [-0.5, -0.5, 0.0]), dtype=np.float64),
            particle_hi=np.array(getattr(args, "emit_hi", [0.5, 0.5, 0.05]), dtype=np.float64),
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


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    import argparse  # noqa: PLC0415

    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1.5, 0.5, 0.05])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -10.0])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=3)

    parser.add_argument("--density", type=float, default=2500.0)
    parser.add_argument("--air-drag", type=float, default=0.0)
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0)
    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e15)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.48)
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

    parser.add_argument("--joint-target-ke", type=float, default=200.0)
    parser.add_argument("--joint-target-kd", type=float, default=100.0)
    parser.add_argument("--ik-iters", type=int, default=50)

    parser.add_argument(
        "--show-foot-traj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show LF/RF foot trajectories",
    )
    parser.add_argument("--show-impulses", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--enable-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show real-time LF interaction force curve (matplotlib)",
    )

    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--gait-period", type=float, default=0.5)
    parser.add_argument("--swing-ratio", type=float, default=0.5)
    parser.add_argument("--step-length", type=float, default=0.4)
    parser.add_argument("--step-height", type=float, default=0.1)
    parser.add_argument("--swing-apex-x1", type=float, default=0.25)
    parser.add_argument("--swing-apex-x2", type=float, default=0.75)
    parser.add_argument("--swing-y0", type=float, default=0.0)
    parser.add_argument("--swing-y1", type=float, default=0.1)
    parser.add_argument("--swing-y2", type=float, default=0.1)
    parser.add_argument("--swing-y3", type=float, default=0.0)

    parser.add_argument(
        "--auto-drop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically lower robot to terrain height on reset",
    )
    parser.add_argument("--drop-clearance", type=float, default=0.002)

    parser.add_argument(
        "--collide-shanks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include shank shapes in particle collisions (in addition to feet)",
    )

    default_urdf = os.path.join(os.path.dirname(__file__), "el_mini", "urdf", "el_mini_allmove.urdf")
    parser.add_argument("--robot-urdf", type=str, default=default_urdf)
    parser.add_argument(
        "--robot-floating",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--robot-base-z", type=float, default=0.2)

    parser.add_argument(
        "--plot-lf-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show LF interaction force curve/print",
    )
    parser.add_argument(
        "--plot-rf-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show RF interaction force curve/print",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    except SystemExit:
        pass

    if hasattr(example, "force_history") and example.force_history:
        print("\nMax LF particle force (N):", float(np.max(example.force_history)))
    if hasattr(example, "rf_force_history") and example.rf_force_history:
        print("Max RF particle force (N):", float(np.max(example.rf_force_history)))
