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

		# ===== 逆运动学 & 三步态 =====
		# 步态频率 (Hz)
		self._gait_freq_hz = float(getattr(options, "gait_freq", 1.0))
		# 摆动相腿抬起高度 (m)
		self._swing_height = float(getattr(options, "swing_height", 0.05))
		# 步幅 (m)
		self._step_length = float(getattr(options, "step_length", 0.08))

		# 连杆长度（m）。对应 C 代码中的 L1/L2/L3（若你的尺寸是 mm，请除以 1000）
		self._ik_l1 = float(getattr(options, "ik_l1", 0.045))
		self._ik_l2 = float(getattr(options, "ik_l2", 0.073))
		self._ik_l3 = float(getattr(options, "ik_l3", 0.115))
		# 足端中心位置（m），在“单腿髋关节坐标系”里指定
		self._ik_foot_x_base = float(getattr(options, "ik_foot_x", 0.0))
		self._ik_foot_y_base = float(getattr(options, "ik_foot_y", 0.0))
		self._ik_foot_z_base = float(getattr(options, "ik_foot_z", 0.6))
		# 可选：给 IK 输出角加偏置（rad），用于匹配 URDF 的零位定义
		self._ik_haa_bias = float(getattr(options, "ik_haa_bias", 0.0))
		self._ik_hfe_bias = float(getattr(options, "ik_hfe_bias", 0.0))
		self._ik_kfe_bias = float(getattr(options, "ik_kfe_bias", 0.0))
		# 左侧三条腿（LF/LM/LB）专用偏置（rad）：用于修正左右镜像造成的“左腿姿态不对”问题
		self._ik_haa_bias_left = float(getattr(options, "ik_haa_bias_left", 0.0))
		self._ik_hfe_bias_left = float(getattr(options, "ik_hfe_bias_left", 0.0))
		self._ik_kfe_bias_left = float(getattr(options, "ik_kfe_bias_left", 0.0))

		self._dof_by_joint = self._build_joint_dof_index_map(self._builder_joint_key, self.model)
		self._target_base = np.array(self._default_target_pos_dof, dtype=np.float64, copy=True)

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

			t = float(self.sim_time) + float(sub) * float(self.sim_dt)
			self._apply_ik_joint_targets(t)

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

	@staticmethod
	def _clamp(x: float, lo: float, hi: float) -> float:
		return float(min(max(float(x), float(lo)), float(hi)))

	def _ik_angles(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
		# 对应你 C 代码：
		# angle1 = atan2(y,x)
		# angle2/3 使用 data = L1 - sqrt(x^2+y^2)
		l1, l2, l3 = float(self._ik_l1), float(self._ik_l2), float(self._ik_l3)
		angle1 = float(np.arctan2(float(y), float(x)))
		r = float(np.sqrt(float(x) * float(x) + float(y) * float(y)))
		data = float(l1 - r)
		den = float(np.sqrt(data * data + float(z) * float(z)))
		if den < 1e-9:
			return None

		a = (l2 * l2 - l3 * l3 + data * data + float(z) * float(z)) / (2.0 * l2 * den)
		b = (l3 * l3 - l2 * l2 + data * data + float(z) * float(z)) / (2.0 * l3 * den)
		a = self._clamp(a, -1.0, 1.0)
		b = self._clamp(b, -1.0, 1.0)

		# C 代码里用 atan(z/data)，这里用 atan2 更稳健
		angle2 = float(np.arccos(a) - np.arctan2(float(z), data))
		angle3 = float(-np.arccos(a) - np.arccos(b))
		return angle1, angle2, angle3

	def _apply_ik_joint_targets(self, t: float) -> None:
		# 对所有关节先用“站姿基准”，再覆盖 6 条腿的 HAA/HFE/KFE
		target = np.array(self._target_base, dtype=np.float64, copy=True)

		# URDF 中左右腿的关节坐标系是镜像的：左腿（L*）的 HAA/HFE 关节 frame 额外绕 x 旋转了 pi，
		# 等效为关节轴向相反。因此需要对左腿的 HAA/HFE 目标角做一次反号映射，
		# 否则会出现“相位被抵消”，看起来不像三步态。
		def side_sign(leg: str) -> float:
			return -1.0 if leg in {"LF", "LM", "LB"} else 1.0

		def side_bias_left(leg: str) -> tuple[float, float, float]:
			if leg in {"LF", "LM", "LB"}:
				return self._ik_haa_bias_left, self._ik_hfe_bias_left, self._ik_kfe_bias_left
			return 0.0, 0.0, 0.0

		# --- 三步态轨迹生成 ---
		phase = 2.0 * float(np.pi) * self._gait_freq_hz * t

		def foot_xz_targets(phase_i: float) -> tuple[float, float]:
			# x: 用 cos 做前后摆动；dx/dt ~ sin(phase)。sin>0 代表“向前”（摆动相）。
			sin_i = float(np.sin(phase_i))
			x_i = self._ik_foot_x_base - self._step_length / 2.0 * float(np.cos(phase_i))
			z_i = self._ik_foot_z_base
			if sin_i > 0.0:
				# 摆动相：z 按正弦曲线抬起（0->peak->0）
				# 这里 IK 的 z 往往“向下为正”（你默认 --ik-foot-z=0.6 也是正数），
				# 因此“抬腿”应当让 z 变小（离髋更近）。
				z_i = z_i - self._swing_height * sin_i
			return x_i, z_i
		# 步态组 1 (LF, RM, LB)
		phase1 = phase
		x_target1, z_target1 = foot_xz_targets(phase1)

		# 步态组 2 (RF, LM, RB)
		phase2 = phase + float(np.pi)
		x_target2, z_target2 = foot_xz_targets(phase2)

		y_left = self._ik_foot_y_base
		y_right = -self._ik_foot_y_base

		# --- 分配到各条腿 ---

		# LF (组 1)
		angles = self._ik_angles(x_target1, y_left, z_target1)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("LF")
			b1, b2, b3 = side_bias_left("LF")
			dof = self._dof_by_joint.get("LF_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("LF_HFE")
			if dof is not None:
				target[dof] = -s * a2 - self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("LF_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

		# RF (组 2)
		angles = self._ik_angles(x_target2, y_right, z_target2)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("RF")
			b1, b2, b3 = side_bias_left("RF")
			dof = self._dof_by_joint.get("RF_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("RF_HFE")
			if dof is not None:
				target[dof] = s * a2 + self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("RF_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

		# LM (组 2)
		angles = self._ik_angles(x_target2, y_left, z_target2)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("LM")
			b1, b2, b3 = side_bias_left("LM")
			dof = self._dof_by_joint.get("LM_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("LM_HFE")
			if dof is not None:
				target[dof] = -s * a2 - self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("LM_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

		# RM (组 1)
		angles = self._ik_angles(x_target1, y_right, z_target1)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("RM")
			b1, b2, b3 = side_bias_left("RM")
			dof = self._dof_by_joint.get("RM_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("RM_HFE")
			if dof is not None:
				target[dof] = s * a2 + self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("RM_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

		# LB (组 1)
		angles = self._ik_angles(x_target1, y_left, z_target1)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("LB")
			b1, b2, b3 = side_bias_left("LB")
			dof = self._dof_by_joint.get("LB_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("LB_HFE")
			if dof is not None:
				target[dof] = -s * a2 - self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("LB_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

		# RB (组 2)
		angles = self._ik_angles(x_target2, y_right, z_target2)
		if angles is not None:
			a1, a2, a3 = angles
			s = side_sign("RB")
			b1, b2, b3 = side_bias_left("RB")
			dof = self._dof_by_joint.get("RB_HAA")
			if dof is not None:
				target[dof] = s * a1 + self._ik_haa_bias + b1
			dof = self._dof_by_joint.get("RB_HFE")
			if dof is not None:
				target[dof] = s * a2 + self._ik_hfe_bias + b2
			dof = self._dof_by_joint.get("RB_KFE")
			if dof is not None:
				target[dof] = a3 + self._ik_kfe_bias + b3

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
	parser.add_argument("--joint-target-ke", type=float, default=50000.0)
	parser.add_argument("--joint-target-kd", type=float, default=500.0)
	parser.add_argument(
		"--auto-drop",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="是否在初始化时自动把机器人下放到地面上（仅在 --robot-floating=true 时生效）",
	)
	parser.add_argument("--drop-clearance", type=float, default=0.002, help="Clearance above ground after auto-drop")
	
	# --- 步态参数 ---
	parser.add_argument("--gait-freq", type=float, default=0.5, help="步态频率 (Hz)")
	parser.add_argument("--swing-height", type=float, default=0.1, help="摆动相腿部抬起高度 (m)")
	parser.add_argument("--step-length", type=float, default=0.06, help="步幅 (m)")

	# --- IK 参数 ---
	parser.add_argument("--ik-l1", type=float, default=0.045, help="单腿连杆 L1（m），对应你 C 代码里的 L1")
	parser.add_argument("--ik-l2", type=float, default=0.073, help="单腿连杆 L2（m），对应你 C 代码里的 L2")
	parser.add_argument("--ik-l3", type=float, default=0.115, help="单腿连杆 L3（m），对应你 C 代码里的 L3")
	parser.add_argument("--ik-foot-x", type=float, default=0.0, help="足端中心 x 坐标 (m)")
	parser.add_argument("--ik-foot-y", type=float, default=0.0, help="足端中心 y 坐标 (m)")
	parser.add_argument("--ik-foot-z", type=float, default=0.6, help="足端中心 z 坐标 (m)")
	parser.add_argument("--ik-haa-bias", type=float, default=0.0, help="HAA 角度偏置（rad），用于匹配 URDF 零位")
	parser.add_argument("--ik-hfe-bias", type=float, default=0.0, help="HFE 角度偏置（rad），用于匹配 URDF 零位")
	parser.add_argument("--ik-kfe-bias", type=float, default=0.0, help="KFE 角度偏置（rad），用于匹配 URDF 零位")
	parser.add_argument("--ik-haa-bias-left", type=float, default=0.0, help="左侧腿（LF/LM/LB）HAA 额外偏置（rad）")
	parser.add_argument("--ik-hfe-bias-left", type=float, default=0.0, help="左侧腿（LF/LM/LB）HFE 额外偏置（rad）")
	parser.add_argument("--ik-kfe-bias-left", type=float, default=0.0, help="左侧腿（LF/LM/LB）KFE 额外偏置（rad）")

	default_urdf = os.path.join(os.path.dirname(__file__), "el_mini", "urdf", "el_mini_allmove.urdf")
	parser.add_argument("--robot-urdf", type=str, default=default_urdf, help="URDF path for the robot")
	parser.add_argument(
		"--robot-floating",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Import robot with a floating base (enables auto-drop)",
	)
	parser.add_argument("--robot-base-z", type=float, default=0.3, help="Initial base height (z) for imported robot")

	viewer, args = newton.examples.init(parser)
	example = Example(viewer, args)
	newton.examples.run(example, args)

