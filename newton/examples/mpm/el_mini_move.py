import os

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
	def __init__(self, viewer, options):
		self.fps = float(getattr(options, "fps", 50.0))
		self.frame_dt = 1.0 / float(self.fps)
		self.sim_time = 0.0
		self.sim_substeps = int(getattr(options, "substeps", 4))
		self.sim_dt = float(self.frame_dt) / float(self.sim_substeps)

		self.viewer = viewer

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

		builder.add_ground_plane()

		this_dir = os.path.dirname(__file__)
		default_urdf = os.path.join(this_dir, "el_mini", "urdf", "el_mini_allmove.urdf")
		robot_urdf = str(getattr(options, "robot_urdf", default_urdf))
		robot_floating = bool(getattr(options, "robot_floating", True))
		base_z = float(getattr(options, "robot_base_z", 1.0))

		builder.add_urdf(
			robot_urdf,
			xform=wp.transform(wp.vec3(0.0, 0.0, float(base_z)), wp.quat_identity()),
			floating=robot_floating,
			enable_self_collisions=False,
			collapse_fixed_joints=True,
			ignore_inertial_definitions=False,
		)

		# finalize 前缓存关节命名，用于把 "LF_HAA" 这类关节名映射到 dof-space 控制索引
		self._builder_joint_key = [str(k) for k in list(getattr(builder, "joint_key", []))]

		self.model = builder.finalize()
		self.model.set_gravity(getattr(options, "gravity", [0.0, 0.0, -10.0]))

		self.state_0 = self.model.state()
		self.state_1 = self.model.state()

		self.control = self.model.control()

		if self.model.joint_target_ke is not None and self.model.joint_target_kd is not None:
			ke = float(getattr(options, "joint_target_ke", 150.0))
			kd = float(getattr(options, "joint_target_kd", 5.0))
			self._set_ke_kd_for_nonfree_joints(self.model, ke=ke, kd=kd)

		self._default_target_pos_dof = self._build_default_target_pos_dof(self.model, self.state_0)
		dof_count = int(self.model.joint_dof_count)
		self.control.joint_target_pos.assign(self._default_target_pos_dof.astype(np.float64).tolist())
		self.control.joint_target_vel.assign(np.zeros(dof_count, dtype=np.float64).tolist())

		# ===== 正弦位控（六条腿逐条写） =====
		self.controller = str(getattr(options, "controller", "sine")).lower()
		if self.controller not in ("stand", "sine"):
			raise ValueError("--controller must be one of: stand, sine")

		self._sine_freq_hz = float(getattr(options, "sine_freq", 1.0))
		# 正弦频率（Hz）。相位：phase = 2π f t
		# 下面 amp/bias 都是“关节目标角”的附加项，单位：弧度(rad)
		self._sine_haa_amp = float(getattr(options, "sine_haa_amp", 0.2))
		self._sine_hfe_amp = float(getattr(options, "sine_hfe_amp", 0.35))
		self._sine_kfe_amp = float(getattr(options, "sine_kfe_amp", 0.3))
		self._sine_haa_bias = float(getattr(options, "sine_haa_bias", 0.0))
		self._sine_hfe_bias = float(getattr(options, "sine_hfe_bias", 0.0))
		self._sine_kfe_bias = float(getattr(options, "sine_kfe_bias", 0.0))

		self._dof_by_joint = self._build_joint_dof_index_map(self._builder_joint_key, self.model)
		self._sine_base = np.array(self._default_target_pos_dof, dtype=np.float64, copy=True)

		newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

		auto_drop = bool(getattr(options, "auto_drop", True))
		if robot_floating and auto_drop:
			clearance = float(getattr(options, "drop_clearance", 0.002))
			self._drop_to_ground(self.model, self.state_0, ground_z=0.0, clearance=clearance)
			self.state_1.joint_q.assign(self.state_0.joint_q.numpy().tolist())
			if self.state_0.joint_qd is not None and self.state_1.joint_qd is not None:
				self.state_1.joint_qd.assign(self.state_0.joint_qd.numpy().tolist())
			newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

		self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
		self.viewer.set_model(self.model)

	def step(self):
		for sub in range(self.sim_substeps):
			self.state_0.clear_forces()
			self.viewer.apply_forces(self.state_0)

			if self.controller == "sine":
				# 用仿真时间保证相位连续
				t = float(self.sim_time) + float(sub) * float(self.sim_dt)
				self._apply_sine_pos_control(t)

			contacts = self.model.collide(self.state_0)
			self.solver.step(self.state_0, self.state_1, control=self.control, contacts=contacts, dt=self.sim_dt)
			self.state_0, self.state_1 = self.state_1, self.state_0

		self.sim_time += self.frame_dt

	def render(self):
		self.viewer.begin_frame(self.sim_time)
		self.viewer.log_state(self.state_0)
		self.viewer.end_frame()

	def test_final(self):
		newton.examples.test_body_state(
			self.model,
			self.state_0,
			"all bodies are above the ground",
			lambda q, qd: q[2] > 0.0,
		)

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

	@classmethod
	def _transform_points(cls, xform: np.ndarray, pts: np.ndarray) -> np.ndarray:
		x = np.asarray(xform, dtype=np.float64).reshape(7)
		p = x[:3]
		q = x[3:]
		return cls._quat_rotate(q, pts) + p

	@staticmethod
	def _build_joint_dof_index_map(joint_key: list[str], model: newton.Model) -> dict[str, int]:
		qd_start = model.joint_qd_start.numpy().astype(np.int32)
		q_start = model.joint_q_start.numpy().astype(np.int32)
		joint_count = int(qd_start.shape[0] - 1)

		out: dict[str, int] = {}
		for jid in range(min(joint_count, len(joint_key))):
			q0, q1 = int(q_start[jid]), int(q_start[jid + 1])
			d0, d1 = int(qd_start[jid]), int(qd_start[jid + 1])
			# 只映射 1-DoF/多 DoF 的常规关节；free joint 的 q_span=7/qd_span=6 不一致，跳过
			if d1 <= d0:
				continue
			if (q1 - q0) != (d1 - d0):
				continue
			out[str(joint_key[jid])] = int(d0)
		return out

	def _apply_sine_pos_control(self, t: float) -> None:
		if self._sine_freq_hz <= 0.0:
			return

		phase = 2.0 * float(np.pi) * float(self._sine_freq_hz) * float(t)
		phase2 = 4.0 * float(np.pi) * float(self._sine_freq_hz) * float(t + self.sim_dt)

		# tripod 相位：显式写出来，不做“先分组再循环”
		s_lf = float(np.sin(phase))
		s_lf2 = float(np.sin(phase2))
		s_rm = float(np.sin(phase))
		s_lh = float(np.sin(phase))
		s_rf = float(np.sin(phase + float(np.pi)))
		s_lm = float(np.sin(phase + float(np.pi)))
		s_rh = float(np.sin(phase + float(np.pi)))

		target = np.array(self._sine_base, dtype=np.float64, copy=True)

		# LF
		dof = self._dof_by_joint.get("LF_HAA")
		target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_lf
		
		dof = self._dof_by_joint.get("LF_HFE")
		if s_lf < 0:
			target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		else:
			target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_lf2

		dof = self._dof_by_joint.get("LF_KFE")
		target[dof] = self._sine_base[dof] + self._sine_kfe_bias + self._sine_kfe_amp * s_lf

		# # RF
		# dof = self._dof_by_joint.get("RF_HAA")		
		# target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_rf
		
		# dof = self._dof_by_joint.get("RF_HFE")		
		# if s_rf < 0:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		# else:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_rf
		
		# # dof = self._dof_by_joint.get("RF_KFE")		
		# # target[dof] = self._sine_base[dof] + self._sine_kfe_bias + self._sine_kfe_amp * s_rf

		# # LM
		# dof = self._dof_by_joint.get("LM_HAA")		
		# target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_lm
		
		# dof = self._dof_by_joint.get("LM_HFE")		
		# if s_lm < 0:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		# else:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_lm
		
		# # dof = self._dof_by_joint.get("LM_KFE")		
		# # target[dof] = self._sine_base[dof] + self._sine_kfe_bias + self._sine_kfe_amp * s_lm

		# # RM
		# dof = self._dof_by_joint.get("RM_HAA")		
		# target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_rm
		
		# dof = self._dof_by_joint.get("RM_HFE")		
		# if s_rm < 0:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		# else:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_rm
		
		# # dof = self._dof_by_joint.get("RM_KFE")		
		# # target[dof] = self._sine_base[dof] + self._sine_kfe_bias + self._sine_kfe_amp * s_rm

		# # LH
		# dof = self._dof_by_joint.get("LB_HAA")	
		# target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_lh
		
		# dof = self._dof_by_joint.get("LB_HFE")		
		# if s_lh < 0:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		# else:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_lh
		
		# # dof = self._dof_by_joint.get("LB_KFE")	
		# # target[dof] = self._sine_base[dof] + self._sine_kfe_bias +  self._sine_kfe_amp * s_lh

		# # RH
		# dof = self._dof_by_joint.get("RB_HAA")
		# target[dof] = self._sine_base[dof] + self._sine_haa_bias + self._sine_haa_amp * s_rh
		
		# dof = self._dof_by_joint.get("RB_HFE")
		# if s_rh < 0:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * 0.0
		# else:
		# 	target[dof] = self._sine_base[dof] + self._sine_hfe_bias + self._sine_hfe_amp * s_rh

		# # dof = self._dof_by_joint.get("RB_KFE")
		# # target[dof] = self._sine_base[dof] + self._sine_kfe_bias + self._sine_kfe_amp * s_rh

		self.control.joint_target_pos.assign(target.astype(np.float64).tolist())
		self.control.joint_target_vel.assign(np.zeros(int(self.model.joint_dof_count), dtype=np.float64).tolist())

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


if __name__ == "__main__":
	parser = newton.examples.create_parser()
	import argparse  # noqa: PLC0415

	parser.add_argument("--fps", type=float, default=50.0)
	parser.add_argument("--substeps", type=int, default=4)
	parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -10.0])
	parser.add_argument("--joint-target-ke", type=float, default=150.0)
	parser.add_argument("--joint-target-kd", type=float, default=5.0)
	parser.add_argument(
		"--auto-drop",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="是否在初始化时自动把机器人下放到地面上（仅在 --robot-floating=true 时生效）",
	)
	parser.add_argument("--drop-clearance", type=float, default=0.002, help="Clearance above ground after auto-drop")
	parser.add_argument(
		"--controller",
		type=str,
		choices=["stand", "sine"],
		default="sine",
		help="关节目标控制模式：stand=锁定初始站姿；sine=在站姿上叠加正弦位控",
	)
	parser.add_argument("--sine-freq", type=float, default=0.5, help="正弦频率（Hz），phase=2π f t")
	parser.add_argument("--sine-haa-amp", type=float, default=0.2, help="HAA 关节正弦幅值（rad）")
	parser.add_argument("--sine-hfe-amp", type=float, default=-0.35, help="HFE 关节正弦幅值（rad）")
	parser.add_argument("--sine-kfe-amp", type=float, default=-0.1, help="KFE 关节正弦幅值（rad）")
	parser.add_argument("--sine-haa-bias", type=float, default=0.0, help="HAA 关节偏置（rad），与站姿目标相加")
	parser.add_argument("--sine-hfe-bias", type=float, default=0.3, help="HFE 关节偏置（rad），与站姿目标相加")
	parser.add_argument("--sine-kfe-bias", type=float, default=0.25, help="KFE 关节偏置（rad），与站姿目标相加")

	default_urdf = os.path.join(os.path.dirname(__file__), "el_mini", "urdf", "el_mini_allmove.urdf")
	parser.add_argument("--robot-urdf", type=str, default=default_urdf, help="URDF path for the robot")
	parser.add_argument(
		"--robot-floating",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Import robot with a floating base (enables auto-drop)",
	)
	parser.add_argument("--robot-base-z", type=float, default=1.0, help="Initial base height (z) for imported robot")

	viewer, args = newton.examples.init(parser)
	example = Example(viewer, args)
	newton.examples.run(example, args)
	
