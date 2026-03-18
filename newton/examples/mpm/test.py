# uv run -p 3.10 -m newton.examples.mpm.test

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

# --- 新增内核 ---
@wp.kernel
def add_forces_kernel(
    target_forces: wp.array(dtype=wp.spatial_vector),
    source_forces: wp.array(dtype=wp.spatial_vector)
):
    """
    将 source_forces 的每个元素加到 target_forces 的对应元素上。
    """
    tid = wp.tid()
    if tid >= target_forces.shape[0]:
        return
    
    # 直接使用 + 操作符，然后赋值。因为每个线程只操作自己的元素，所以这是安全的。
    target_forces[tid] = target_forces[tid] + source_forces[tid]

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

        # ========== 物块系统构建 ==========
        body_block_extents = (0.2, 0.2, 0.1)  # 物块尺寸
        hx, hy, hz = body_block_extents
        radius =0.15

        # 构建刚体沙子的耦合系统
        drop_z = 2.0
        mass = 10.0 
        
        # 创建三个物块，位于不同位置
        self.body_block1 = builder.add_body(xform=wp.transform(p=wp.vec3(-0.5, 0.0, drop_z), q=wp.quat_identity()))
        self.body_block2 = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        self.body_block3 = builder.add_body(xform=wp.transform(p=wp.vec3(0.5, 0.0, drop_z), q=wp.quat_identity()))

        # shape_cfg = newton.ModelBuilder.ShapeConfig(density=mass / (8 * hx * hy * hz))
        # builder.add_shape_box(self.body_block1, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
        # builder.add_shape_box(self.body_block2, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)
        # builder.add_shape_box(self.body_block3, hx=hx, hy=hy, hz=hz, cfg=shape_cfg)

        shape_cfg = newton.ModelBuilder.ShapeConfig(density=mass / (4/3 *  np.pi * radius**3) )
        builder.add_shape_sphere(self.body_block1, radius=radius, cfg=shape_cfg)
        builder.add_shape_sphere(self.body_block2, radius=radius, cfg=shape_cfg)
        builder.add_shape_sphere(self.body_block3, radius=radius, cfg=shape_cfg)

        # ========== 沙子模型构建 ==========
        mpm_builder = newton.ModelBuilder()
        # 沙子相关：在指定区域生成 MPM 粒子
        sand_particles, snow_particles, mud_particles, block_particles, particle_weights = Example.emit_particles(
            mpm_builder, options
        )

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

        # ========== 连续混合材料参数（替代“阈值分材料”） ==========
        # 这里的核心思路：
        # - `emit_particles()` 会为每个粒子生成连续权重 weights=[sand,snow,mud,block]
        # - 物性不再“按类别硬切换”，而是对每个粒子按权重连续插值（材料边界自然渐变）
        # - 这种做法能避免分位数/硬阈值造成的“刻意分区”，也更利于做“混杂地形”的过渡带
        #
        # 参数混合的小经验：
        # - 摩擦/硬化等通常在线性域混合即可（同量纲、变化不跨数量级）
        # - 屈服压力等跨数量级很大的参数，建议在 log 域混合，避免被最大值“淹没”
        #
        # 注意：MaterialParameters 是 Warp struct，字段是 wp.array，不能用 `[:] =` 切片赋值，需用 `assign()` 写回。
        mp = mpm_model.material_parameters
        w = particle_weights.astype(np.float64, copy=False)
        w_sand, w_snow, w_mud, w_blk = w[:, 0], w[:, 1], w[:, 2], w[:, 3]

        def _blend_lin(a: float, b: float, c: float, d: float) -> np.ndarray:
            # 线性混合：适合摩擦、硬化等“同量纲、变化范围不跨数量级”的参数
            return w_sand * a + w_snow * b + w_mud * c + w_blk * d

        def _blend_log10(a: float, b: float, c: float, d: float) -> np.ndarray:
            # 对数域混合：适合屈服压力这类跨多个数量级的参数。
            # 直接线性混合会被大值“淹没”，用 log10 先混合再还原更合理。
            eps = 1.0e-30
            la = float(np.log10(max(a, eps)))
            lb = float(np.log10(max(b, eps)))
            lc = float(np.log10(max(c, eps)))
            ld = float(np.log10(max(d, eps)))
            return np.power(10.0, _blend_lin(la, lb, lc, ld))

        yield_pressure = _blend_log10(1.0e5, 2.0e4, 1.0e10, 1.0e8)
        yield_stress = _blend_lin(0.0, 1.0e3, 3.0e2, 1.0e7)
        tensile_yield_ratio = _blend_lin(0.0, 0.05, 1.0, 0.8)
        hardening = _blend_lin(0.0, 10.0, 2.0, 20.0)
        # 摩擦系数物理上通常在 [0,1] 内，混合后做一次截断更稳妥。
        friction = np.clip(_blend_lin(0.48, 0.10, 0.00, 0.60), 0.0, 1.0)

        wp_device = self.mpm_model.device
        # 重要：mp.* 是 wp.array，必须用 assign 写回（并且 device 要与模型一致）。
        mp.yield_pressure.assign(wp.array(yield_pressure, dtype=float, device=wp_device))
        mp.yield_stress.assign(wp.array(yield_stress, dtype=float, device=wp_device))
        mp.tensile_yield_ratio.assign(wp.array(tensile_yield_ratio, dtype=float, device=wp_device))
        mp.hardening.assign(wp.array(hardening, dtype=float, device=wp_device))
        mp.friction.assign(wp.array(friction, dtype=float, device=wp_device))

        # 通知MPM模型，粒子材质参数已发生变化
        mpm_model.notify_particle_material_changed()
        
        # 从刚体模型而不是沙子模型中读取碰撞体
        mpm_model.setup_collider(model=self.model)

        # ========== 求解器初始化 ==========
        # 物块求解器
        self.solver_block = newton.solvers.SolverXPBD(self.model)

        # 沙子 MPM 求解器
        self.solver_mpm = SolverImplicitMPM(mpm_model, mpm_options)

        # ========== 仿真状态初始化 ==========        
        # 沙子相关：创建双缓冲状态（用于时间步进）
        self.state_0 = self.model.state()  # 当前状态
        self.state_1 = self.model.state()  # 下一状态

        self.mpm_state_0 = self.mpm_model.state()
        # 沙子相关：为状态添加 MPM 专用字段（网格信息、粒子缓存等）
        self.solver_mpm.enrich_state(self.mpm_state_0)

        # 初始化机器人正向运动学和碰撞网格
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

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
        self.body_mpm_forces = wp.zeros_like(self.state_0.body_f)

        # 颜色也做连续混合（更直观看到“渐变/混杂”）
        # 说明：这里的颜色仅用于可视化，真正的物性由上面的 material_parameters 控制。
        cols = np.array(
            [
                [0.7, 0.6, 0.4],   # sand
                [0.75, 0.75, 0.8], # snow
                [0.4, 0.25, 0.25], # mud
                [0.5, 0.5, 0.5],   # block
            ],
            dtype=np.float32,
        )
        colors_np = (particle_weights.astype(np.float32) @ cols).astype(np.float32)
        self.particle_colors = wp.array(colors_np, dtype=wp.vec3, device=self.mpm_model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True  # 显示沙子粒子
        self.show_impulses = False

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # ========== F-t 曲线绘制设置 ==========
        self.force_history = [[], [], []]  # 对应三个物块
        self.time_history = []
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()
        self.lines = [
            self.ax.plot([], [], label='Block 1 (Left)')[0],
            self.ax.plot([], [], label='Block 2 (Center)')[0],
            self.ax.plot([], [], label='Block 3 (Right)')[0]
        ]
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Force (N)")
        self.ax.set_title("Interaction Force vs. Time")
        self.ax.legend()
        self.ax.grid(True)
        self.capture()

    def capture(self):
        """设置CUDA图优化以加速仿真"""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_block()
            self.graph = capture.graph
        
        self.mpm_graph = None
        if wp.get_device().is_cuda and self.solver_mpm.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_mpm()
            self.mpm_graph = capture.graph

    def collect_collider_impulses(self):
        """从MPM求解器收集碰撞冲量"""
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.solver_mpm.collect_collider_impulses(
            self.mpm_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    def simulate_block(self):
        """物块仿真：执行物块动力学计算"""
        for _ in range(self.sim_substeps):

            self.state_0.clear_forces()  # 物块相关：清除上一步的外力

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
            self.body_mpm_forces.assign(self.state_0.body_f)

            self.viewer.apply_forces(self.state_0)  # 应用用户交互力

            # --- 直接在主模型上进行碰撞检测 ---
            self.contacts = self.model.collide(self.state_0)
            
            # 物块求解器步进
            self.solver_block.step(self.state_0, self.state_1, control=None, contacts=self.contacts, dt=self.sim_dt)

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
                    self.body_mpm_forces,
                    self.model.body_inv_inertia,
                    self.model.body_inv_mass,
                    self.mpm_state_0.body_q, # 结果直接写回 mpm_state_0
                    self.mpm_state_0.body_qd,
                ],
            )

        # 沙子相关：MPM 求解器主要步骤
        # 1. 粒子到网格传输（P2G）：将粒子质量/动量汇总到网格点
        # 2. 网格动力学求解：在网格上计算内力、外力，求解速度更新
        # 3. 网格到粒子传输（G2P）：将网格速度插值回粒子
        # 4. 粒子位置更新和材料状态更新
        self.solver_mpm.step(self.mpm_state_0, self.mpm_state_0, contacts=None, control=None, dt=self.frame_dt)

        # 保存冲量以便施加回刚体
        self.collect_collider_impulses()

    def simulate(self):
        """主仿真步骤：每帧调用一次"""
        # ========== 执行仿真步骤 ==========
        # 1. 执行物块仿真
        self.simulate_block()

        # 2. 执行沙子仿真
        self.simulate_mpm()

    
    def step(self):
        """设置CUDA图优化以加速仿真"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_block()

        if self.mpm_graph:
            wp.capture_launch(self.mpm_graph)
        else:
            self.simulate_mpm()

        self.sim_time += self.frame_dt
        
        # ========== 计算并记录总作用力 ==========
        forces = self.body_mpm_forces.numpy()
        force_magnitudes = [
            np.linalg.norm(forces[self.body_block1][:3]),
            np.linalg.norm(forces[self.body_block2][:3]),
            np.linalg.norm(forces[self.body_block3][:3])
        ]

        for i in range(3):
            self.force_history[i].append(force_magnitudes[i])
        self.time_history.append(self.sim_time)

    def render(self):
        """渲染函数：显示沙子粒子和障碍物"""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)  # 记录当前沙子状态用于可视化
        self.viewer.log_points(
            name="/model/particles",
            points=self.mpm_state_0.particle_q,
            radii=self.mpm_model.particle_radius,
            colors=self.particle_colors,
            hidden=False,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.solver_mpm.collect_collider_impulses(self.mpm_state_0)
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
        if self.time_history:
            forces_str = " | ".join([f"Block {i+1}: {self.force_history[i][-1]:8.2f} N" for i in range(3)])
            print(f"\rInteraction Forces: {forces_str}", end="")

            # 更新曲线数据
            for i in range(3):
                self.lines[i].set_xdata(self.time_history)
                self.lines[i].set_ydata(self.force_history[i])
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def render_ui(self, imgui):
        """渲染UI界面"""
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """根据 `perlin_noise.md` 的建议生成“混杂地形”。

        - 多尺度 Perlin（low/mid/high）叠加，得到连续的混合信号
        - 不再用分位数/硬阈值“分材料”，而是用连续权重混合物理属性
        - 可选：用高频噪声对粒子初始位置做小扰动，打破规则网格

        Returns:
            (sand_ids, snow_ids, mud_ids, block_ids, weights):
                - *_ids: 为可视化/调试提供的“主导材料”粒子集合（argmax 权重）
                - weights: (N,4) float32，按粒子 id 对齐的连续权重 [sand, snow, mud, block]

        使用说明（命令行）：
        - 更“精细/颗粒感”：提高 `--noise-frequency` 或 `--noise-octaves`
        - 更“大块分区/宏观变化”：降低 `--noise-frequency` 或调小 `--noise-low-frequency-mult`
        - 改一张“地形图样”（可复现）：修改 `--noise-seed`
        - 石块稀疏度：调 `--noise-block-threshold`（越大越少，建议 0.8~0.95）
        - 打破网格条纹：调 `--noise-position-jitter`（单位 voxel-size，建议 0.0~0.5）

        性能提示：当前实现是在 Python for-loop 中对每个粒子采样多次 Perlin（low/mid/high + jitter），
        `--noise-octaves`、粒子数、以及 high 频采样都会显著影响运行时间。
        """

        def _clamp01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        def _smoothstep(edge0: float, edge1: float, x: float) -> float:
            # 平滑阶跃：让“阈值”变成连续过渡（边界不生硬）。
            # 这里用 3t^2-2t^3 的经典 smoothstep。
            if edge0 == edge1:
                return 0.0
            t = _clamp01((x - edge0) / (edge1 - edge0))
            return t * t * (3.0 - 2.0 * t)

        def _tri(x: float, center: float, width: float) -> float:
            # 三角形权重（1D membership）：
            # - x 接近 center 时权重最大
            # - 距离超过 width 后降到 0
            # 用于把“中频噪声 n_mid”映射为 sand/snow/mud 的连续比例。
            if width <= 0.0:
                return 0.0
            return max(0.0, 1.0 - abs(x - center) / width)

        def _noise01(p, freq: float, base: int) -> float:
            # ========== 3D Perlin 噪声采样 ==========
            # - 输入是世界坐标 p（乘以频率 freq），不同 freq 对应不同“尺度”
            # - pnoise3 返回 [-1, 1]，这里映射到 [0, 1]
            # - octaves/persistence/lacunarity 形成类 fBm 叠加：octaves 越大越“精细”，但越慢
            # - base=seed：用于可复现地改变噪声图样（同一参数每次运行一致）
            n = noise.pnoise3(
                float(p[0]) * freq,
                float(p[1]) * freq,
                float(p[2]) * freq,
                octaves=int(getattr(args, "noise_octaves", 4)),
                persistence=float(getattr(args, "noise_persistence", 0.5)),
                lacunarity=float(getattr(args, "noise_lacunarity", 2.0)),
                base=int(base),
            )
            return 0.5 * (float(n) + 1.0)

        # 1) 生成所有粒子
        # 注意：这里先按一个“占位密度”生成粒子；后面会按每粒子权重混合密度，覆盖写回 particle_mass。
        all_ids = Example._spawn_particles(
            builder,
            args,
            particle_lo=np.array([-2.0, -2.0, 0.0]),
            particle_hi=np.array([2.0, 2.0, 0.25]),
            density=1000,  # 初始占位，后面会按混合密度覆盖到 particle_mass
            flags=newton.ParticleFlags.ACTIVE,
        )

        # ========== 2) 多尺度 Perlin：low/mid/high ==========
        # 经验用法：
        # - low：大尺度区域（决定“泥区/干区”等宏观分布；也可用来调“压实度”）
        # - mid：主导材料混合（沙/雪/泥的主要变化尺度）
        # - high：微细节（边界破碎、颗粒感），通常权重更小，避免“噪声满屏”
        #
        # 这里将三层噪声按权重线性叠加得到 terrain：
        #   terrain = w_low*n_low + w_mid*n_mid + w_high*n_high
        # 其中 n_* 都在 [0,1]，方便后续用 smoothstep/tri 等连续函数映射到材料权重。
        base_freq = float(getattr(args, "noise_frequency", 0.0))
        low_mult = float(getattr(args, "noise_low_frequency_mult", 0.0))
        high_mult = float(getattr(args, "noise_high_frequency_mult", 0.0))
        w_low, w_mid, w_high = 0.6, 0.3, 0.1
        base_seed = int(getattr(args, "noise_seed", 0))
        block_threshold = float(getattr(args, "noise_block_threshold", 0.0))

        # ========== 可选：粒子位置扰动（打破规则网格） ==========
        # - 单位：voxel_size（即 dx）
        # - 建议范围：0.0~0.5（过大可能引入初始穿插/局部过密，导致数值不稳）
        pos_jitter = float(getattr(args, "noise_position_jitter", 0.0))
        voxel_size = float(getattr(args, "voxel_size", 0.05))

        particle_count = len(builder.particle_q)
        weights = np.zeros((particle_count, 4), dtype=np.float32)

        # 参考值（与你当前代码一致）
        rho_sand, rho_snow, rho_mud, rho_block = 2500.0, 300.0, 1000.0, 2700.0

        for pid in all_ids:
            p = builder.particle_q[pid]

            n_low = _noise01(p, base_freq * low_mult, base_seed + 11)
            n_mid = _noise01(p, base_freq * 1.0, base_seed + 23)
            n_high = _noise01(p, base_freq * high_mult, base_seed + 37)

            # terrain 是一个“连续混合信号”，用于辅助生成 block 等稀疏结构
            terrain = w_low * n_low + w_mid * n_mid + w_high * n_high

            # “湿/黏”区域：用低频决定大范围泥区（边界会自然模糊）
            mudness = _smoothstep(0.55, 0.90, n_low)

            # 材料比例：用中频做连续混合，不再硬分类
            # - 这里的三个中心值 (0.2/0.5/0.8) 是经验选取：把 [0,1] 分成三段但保持连续
            # - mud 额外受 mudness 抬高，形成“大片泥区 + 局部变化”
            w_sand = _tri(n_mid, 0.20, 0.25)
            w_mud = max(_tri(n_mid, 0.50, 0.25), mudness)
            w_snow = _tri(n_mid, 0.80, 0.25) * (1.0 - 0.7 * mudness)

            # 岩石/石块：让它稀疏出现（可通过阈值调稀疏程度）
            # - threshold 越大：越“难触发”，block 越少
            # - 使用 smoothstep：从“开始出现”到“完全 block”有渐变，而不是硬切
            w_blk = _smoothstep(block_threshold, 1.0, terrain)

            # 给高频一点点影响，让边界更自然（但不至于“噪声满屏”）
            detail = (n_high - 0.5) * 0.10
            w_sand = max(0.0, w_sand + detail)
            w_snow = max(0.0, w_snow - detail)

            # 归一化：保证每个粒子的权重和为 1
            # - 先归一化 sand/snow/mud
            # - 再用 (1-w_blk) 给 block 让出比例
            # - 最后整体再归一化一次，避免数值误差
            s = w_sand + w_snow + w_mud
            if s <= 1.0e-8:
                w_sand, w_snow, w_mud = 1.0, 0.0, 0.0
                s = 1.0
            w_sand, w_snow, w_mud = w_sand / s, w_snow / s, w_mud / s

            w_sand *= (1.0 - w_blk)
            w_snow *= (1.0 - w_blk)
            w_mud *= (1.0 - w_blk)

            s2 = w_sand + w_snow + w_mud + w_blk
            if s2 <= 1.0e-8:
                w_sand, w_snow, w_mud, w_blk = 1.0, 0.0, 0.0, 0.0
                s2 = 1.0
            w_sand, w_snow, w_mud, w_blk = w_sand / s2, w_snow / s2, w_mud / s2, w_blk / s2

            weights[pid, 0] = w_sand
            weights[pid, 1] = w_snow
            weights[pid, 2] = w_mud
            weights[pid, 3] = w_blk

            # 连续密度混合 + 低频压实度（更像“压实/湿区”）
            # - rho_* 是“参考密度”（可理解为不同材料的目标密度）
            # - compaction 用 low 噪声调一个轻微倍率，让大尺度区域“更紧/更松”
            rho = (
                w_sand * rho_sand
                + w_snow * rho_snow
                + w_mud * rho_mud
                + w_blk * rho_block
            )
            compaction = 0.85 + 0.30 * n_low  # [0.85, 1.15]
            rho *= compaction

            # 写回 particle_mass（mass = density * volume）
            # 说明：MPM 里粒子“质量”是核心输入；我们通过密度混合来实现“不同物性”的第一步。
            r = float(builder.particle_radius[pid])
            volume = (4.0 / 3.0) * np.pi * (r ** 3)
            builder.particle_mass[pid] = float(rho * volume)

            # 可选：基于高频噪声扰动初始位置（单位：voxel_size）
            # 目的：粒子初始来自规则网格 + jitter，仍可能有可见条纹；再叠加一点“结构化噪声”更自然。
            if pos_jitter > 0.0:
                jx = _noise01(p, base_freq * high_mult, base_seed + 101) - 0.5
                jy = _noise01(p, base_freq * high_mult, base_seed + 103) - 0.5
                jz = _noise01(p, base_freq * high_mult, base_seed + 107) - 0.5
                builder.particle_q[pid] = wp.vec3(
                    float(p[0]) + voxel_size * pos_jitter * float(jx),
                    float(p[1]) + voxel_size * pos_jitter * float(jy),
                    float(p[2]) + voxel_size * pos_jitter * float(jz),
                )

        # 3) 为可视化提供“主导材料”集合（argmax 权重）
        # 注意：这些集合只用于 debug/可视化；真正的材料行为由连续 weights 控制。
        dominant = np.argmax(weights[all_ids], axis=1)
        sand_ids = all_ids[dominant == 0]
        snow_ids = all_ids[dominant == 1]
        mud_ids = all_ids[dominant == 2]
        block_ids = all_ids[dominant == 3]

        return (
            np.array(sand_ids, dtype=int),
            np.array(snow_ids, dtype=int),
            np.array(mud_ids, dtype=int),
            np.array(block_ids, dtype=int),
            weights,
        )

    # 兼容：旧版“按类别设置密度”的函数仍保留（目前不再使用）
    @staticmethod
    def _assign_density(builder, particle_ids, density):
        for pid in particle_ids:
            radius = builder.particle_radius[pid]
            volume = (4.0 / 3.0) * np.pi * radius**3
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
    # 交互相关：粒子类型选择
    parser.add_argument("--particle-type", default="sand", choices=["sand", "snow", "mud"], type=str)
    # 沙子相关：粒子生成区域
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.2])
    # 沙子相关：重力设置
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    # 仿真控制：帧率和子步
    parser.add_argument("--fps", type=float, default=160)
    parser.add_argument("--substeps", type=int, default=16)

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
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)        # 网格体素大小

    # ========== 颗粒分区噪声参数（Perlin） ==========
    # 这些参数控制 `emit_particles()` 里 low/mid/high 的噪声采样。
    # 使用建议：
    # - 想更“精细”：提高 `--noise-frequency` 或 `--noise-octaves`
    # - 想换一个地形图样：改 `--noise-seed`
    # 性能提示：octaves 会显著增加 Perlin 计算量（这里是按粒子循环采样）。

    # frequency 越大，噪声变化越密（更精细）
    parser.add_argument("--noise-frequency", type=float, default=0.0)
    # octaves 越大，叠加细节越多（更精细，但更慢）
    parser.add_argument("--noise-octaves", type=int, default=1)
    # persistence 越大，高频细节权重越高（更“粗糙/颗粒感”）
    parser.add_argument("--noise-persistence", type=float, default=0.5)
    # lacunarity 越大，每层 octave 的频率增长越快（更快进入细节）
    parser.add_argument("--noise-lacunarity", type=float, default=1.0)
    # seed/base：改变噪声图样（不同随机地形）
    parser.add_argument("--noise-seed", type=int, default=0)

    # ========== 混杂地形：多尺度控制 ==========
    # low/mid/high 采样频率分别是：
    # - low = noise-frequency * noise-low-frequency-mult
    # - mid = noise-frequency
    # - high = noise-frequency * noise-high-frequency-mult
    # 一般 low<1, high>1。
    parser.add_argument("--noise-low-frequency-mult", type=float, default=0.2)
    parser.add_argument("--noise-high-frequency-mult", type=float, default=4.0)
    # 石块稀疏度：越大越少（建议 0.8~0.95）
    parser.add_argument("--noise-block-threshold", type=float, default=0.85)
    # 粒子初始位置噪声扰动（单位：voxel-size；0 关闭，建议 0.0~0.5）
    parser.add_argument("--noise-position-jitter", type=float, default=0.0)

    # 解析参数并初始化查看器
    viewer, args = newton.examples.init(parser)

    # 创建示例并运行
    example = Example(viewer, args)
    newton.examples.run(example, args)


# ==================== API：地形生成器类 ====================
class TerrainGenerator:
    """
    封装的地形生成器，用于在其他文件中调用。
    生成混合颗粒地形（沙子、雪、泥等）并返回 MPM 模型和求解器。
    """
    
    def __init__(self, 
                 emit_lo=[-0.5, -0.5, 0.0],
                 emit_hi=[0.5, 0.5, 0.2],
                 gravity=[0, 0, -10],
                 density=2500.0,
                 friction_coeff=0.48,
                 voxel_size=0.05):
        """
        初始化地形生成器参数
        
        Args:
            emit_lo: 颗粒生成区域下界 [x, y, z]
            emit_hi: 颗粒生成区域上界 [x, y, z]
            gravity: 重力向量 [gx, gy, gz]
            density: 颗粒密度
            friction_coeff: 摩擦系数
            voxel_size: MPM 网格体素大小
        """
        self.emit_lo = np.array(emit_lo)
        self.emit_hi = np.array(emit_hi)
        self.gravity = gravity
        self.density = density
        self.friction_coeff = friction_coeff
        self.voxel_size = voxel_size
        
        # 创建默认参数对象
        self.args = self._create_default_args()
        
    def _create_default_args(self):
        """创建默认的参数配置"""
        class Args:
            pass
        
        args = Args()
        args.voxel_size = self.voxel_size
        args.density = self.density
        args.friction_coeff = self.friction_coeff
        args.young_modulus = 1.0e15
        args.poisson_ratio = 0.3
        args.yield_pressure = 1.0e5
        args.yield_stress = 0.0
        args.tensile_yield_ratio = 0.0
        args.hardening = 0.0
        args.damping = 0.0
        args.air_drag = 1.0
        args.critical_fraction = 0.0
        args.grid_type = "sparse"
        args.solver = "gauss-seidel"
        args.transfer_scheme = "pic"
        args.strain_basis = "P0"
        args.max_iterations = 50
        args.tolerance = 1.0e-6

        # 噪声分区参数（与命令行保持一致的字段名）
        # 说明：`emit_particles()` 通过 getattr(args, ...) 读取这些字段。
        # TerrainGenerator 不是命令行入口，所以这里要补齐同名字段，保持行为一致。
        args.noise_frequency = 1.2
        args.noise_octaves = 4
        args.noise_persistence = 0.5
        args.noise_lacunarity = 2.0
        args.noise_seed = 0

        # 混杂地形：多尺度控制（含 block 稀疏阈值、位置扰动）
        args.noise_low_frequency_mult = 0.2
        args.noise_high_frequency_mult = 4.0
        args.noise_block_threshold = 0.85
        args.noise_position_jitter = 0.0
        
        return args
    
    def generate_terrain(self, collider_model=None):
        """
        生成混合颗粒地形
        
        Args:
            collider_model: 可选的刚体模型，用于设置碰撞体
            
        Returns:
            (mpm_model, solver_mpm, particle_colors): 
                - mpm_model: Newton MPM 模型
                - solver_mpm: MPM 求解器
                - particle_colors: 颗粒颜色数组（用于可视化）
        """
        # 创建 MPM 模型构建器
        mpm_builder = newton.ModelBuilder()
        
        # 生成颗粒并分类
        # 注意：这里返回的 *_particles 主要用于 debug/可视化；物性使用 particle_weights 做连续混合。
        sand_particles, snow_particles, mud_particles, block_particles, particle_weights = Example.emit_particles(
            mpm_builder, self.args
        )
        
        # 最终化 MPM 模型
        mpm_model = mpm_builder.finalize()
        mpm_model.particle_mu = self.friction_coeff
        mpm_model.particle_kd = 0.0
        mpm_model.particle_ke = 1.0e15
        mpm_model.set_gravity(self.gravity)
        
        # 将颗粒列表转换为 warp 数组
        sand_particles = wp.array(sand_particles, dtype=int, device=mpm_model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=mpm_model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=mpm_model.device)
        block_particles = wp.array(block_particles, dtype=int, device=mpm_model.device)
        
        # 创建 MPM 求解器
        mpm_options = SolverImplicitMPM.Options()
        for key in vars(self.args):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(self.args, key))
        
        mpm_model_solver = SolverImplicitMPM.Model(mpm_model, mpm_options)
        
        # 连续混合材料参数（同 Example；Warp wp.array 用 assign 写回）
        # 说明：particle_weights 是 numpy (CPU) 计算结果，这里再写回到 Warp 的 material_parameters。
        mp = mpm_model_solver.material_parameters
        w = particle_weights.astype(np.float64, copy=False)
        w_sand, w_snow, w_mud, w_blk = w[:, 0], w[:, 1], w[:, 2], w[:, 3]

        def _blend_lin(a: float, b: float, c: float, d: float) -> np.ndarray:
            return w_sand * a + w_snow * b + w_mud * c + w_blk * d

        def _blend_log10(a: float, b: float, c: float, d: float) -> np.ndarray:
            eps = 1.0e-30
            la = float(np.log10(max(a, eps)))
            lb = float(np.log10(max(b, eps)))
            lc = float(np.log10(max(c, eps)))
            ld = float(np.log10(max(d, eps)))
            return np.power(10.0, _blend_lin(la, lb, lc, ld))

        yield_pressure = _blend_log10(1.0e5, 2.0e4, 1.0e10, 1.0e8)
        yield_stress = _blend_lin(0.0, 1.0e3, 3.0e2, 1.0e7)
        tensile_yield_ratio = _blend_lin(0.0, 0.05, 1.0, 0.8)
        hardening = _blend_lin(0.0, 10.0, 2.0, 20.0)
        friction = np.clip(_blend_lin(0.48, 0.10, 0.00, 0.60), 0.0, 1.0)

        wp_device = mpm_model.device
        mp.yield_pressure.assign(wp.array(yield_pressure, dtype=float, device=wp_device))
        mp.yield_stress.assign(wp.array(yield_stress, dtype=float, device=wp_device))
        mp.tensile_yield_ratio.assign(wp.array(tensile_yield_ratio, dtype=float, device=wp_device))
        mp.hardening.assign(wp.array(hardening, dtype=float, device=wp_device))
        mp.friction.assign(wp.array(friction, dtype=float, device=wp_device))
        
        # 通知模型材质参数已更改
        mpm_model_solver.notify_particle_material_changed()
        
        # 如果提供了碰撞体模型，设置碰撞体
        if collider_model is not None:
            mpm_model_solver.setup_collider(model=collider_model)
        
        # 创建求解器
        solver_mpm = SolverImplicitMPM(mpm_model_solver, mpm_options)
        
        # 创建颗粒颜色数组（连续混合）
        cols = np.array(
            [
                [0.7, 0.6, 0.4],   # sand
                [0.75, 0.75, 0.8], # snow
                [0.4, 0.25, 0.25], # mud
                [0.5, 0.5, 0.5],   # block
            ],
            dtype=np.float32,
        )
        colors_np = (particle_weights.astype(np.float32) @ cols).astype(np.float32)
        particle_colors = wp.array(colors_np, dtype=wp.vec3, device=mpm_model.device)
        
        return mpm_model, solver_mpm, particle_colors
    
    def get_force_at_position(self, position_2d, mpm_state):
        """
        获取指定 2D 位置处的接触力（法向力和切向力）
        
        Args:
            position_2d: [x, y] 2D 位置坐标
            mpm_state: MPM 状态对象
            
        Returns:
            (F_N, F_T): 法向力和切向力
        """
        # 将 2D 位置转换为 3D（假设 z=0）
        pos_3d = wp.vec3(position_2d[0], position_2d[1], 0.0)
        
        # 这里需要根据实际的 MPM 状态来计算力
        # 简化实现：返回基于距离颗粒的近似力
        # 实际应用中需要查询 MPM 网格或粒子来计算真实接触力
        
        # TODO: 实现真实的力计算逻辑
        # 暂时返回占位值
        F_N = 0.0
        F_T = 0.0
        
        return F_N, F_T