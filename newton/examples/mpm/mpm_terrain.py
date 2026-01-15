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
        self.headless = (viewer is None)  # 无头模式标志

        # ========== 沙子-物块交互系统构建 ==========
        builder = newton.ModelBuilder()

        # 交互相关：添加地面作为沙子的底部边界
        ground_shape = builder.add_ground_plane()

        # ========== 物块系统构建 ==========
        radius = 0.15

        # 构建刚体沙子的耦合系统
        drop_z = 0.7
        mass = 20.0
        self.body_block = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=mass/(4/3 * np.pi * radius**3))
        block_shape = builder.add_shape_sphere(self.body_block, radius=radius, cfg=shape_cfg)

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

        self.particle_colors = wp.full(
            self.mpm_model.particle_count, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.mpm_model.device
        )
        self.particle_colors[sand_particles].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_colors[snow_particles].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_colors[mud_particles].fill_(wp.vec3(0.4, 0.25, 0.25))
        self.particle_colors[block_particles].fill_(wp.vec3(0.5, 0.5, 0.5))

        if not self.headless:
            self.viewer.set_model(self.model)
            self.viewer.show_particles = True  # 显示沙子粒子
        self.show_impulses = False

        if self.viewer is not None and isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        # ========== F-t 曲线绘制设置 ==========
        if not self.headless:
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

            if not self.headless:
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
        # body_mpm_forces 包含线性和角力，我们只关心线性力（前3个分量）
        #           |力和力矩              |复制   |索引             |前三个
        force_vec = self.body_mpm_forces.numpy()[self.body_block][: 3]
        # 这个三维向量的欧几里得范数（也就是它的模长或大小）
        force_magnitude = np.linalg.norm(force_vec)
        
        if not self.headless:
            self.force_history.append(force_magnitude)
            self.time_history.append(self.sim_time)

    def render(self):
        """渲染函数：显示沙子粒子和障碍物"""
        if self.headless:
            return  # 无头模式下不渲染
            
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

    # ==================== 外部 API 接口 ====================
    
    def get_contact_forces(self):
        """
        获取当前物块受到的接触力（用于强化学习环境）。
        返回:
            F_N: 法向力（Z方向分量）
            F_T: 切向力（XY平面内的力的模长）
        """
        # 从 body_mpm_forces 中提取力分量
        force_vec = self.body_mpm_forces.numpy()[self.body_block][:3]
        
        # 法向力：Z方向分量的绝对值
        F_N = abs(force_vec[2])
        
        # 切向力：XY平面内的力的模长
        F_T = np.linalg.norm(force_vec[:2])
        
        return F_N, F_T
    
    def set_block_position(self, x, y, z=None):
        """
        设置物块的位置（用于强化学习环境控制物块移动）。
        参数:
            x, y: 水平位置坐标（2D）
            z: 垂直位置（如果为None，保持当前高度）
        """
        # 将2D坐标映射到3D世界坐标
        # 假设env1的坐标系为 [0, 400] x [0, 400]
        # 映射到 Newton 坐标系 [-2, 2] x [-2, 2]
        world_x = (x - 200) / 50.0  # 中心化并缩放
        world_y = (y - 200) / 50.0
        
        if z is None:
            # 保持当前Z坐标
            current_pos = self.state_0.body_q.numpy()[self.body_block]
            world_z = current_pos[2]
        else:
            world_z = z
        
        # 创建新的变换
        new_transform = wp.transform(
            p=wp.vec3(world_x, world_y, world_z),
            q=wp.quat_identity()
        )
        
        # 更新物块位置 - 使用 numpy 视图修改
        body_q_np = self.state_0.body_q.numpy()
        body_q_np[self.body_block] = new_transform
        self.state_0.body_q.assign(body_q_np)
        
        # 重新计算正向运动学
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
    
    def get_block_position(self):
        """
        获取物块的当前位置。
        返回:
            (x, y, z): 3D坐标
        """
        pos = self.state_0.body_q.numpy()[self.body_block][:3]
        return pos[0], pos[1], pos[2]

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """根据噪声函数在指定区域生成三种不同类型的粒子：沙、雪、泥"""
        # 1. test 原始方法：生成所有粒子
        all_ids = Example._spawn_particles(builder, args,
                                           particle_lo=np.array([-2.0, -2.0, 0.0]),
                                           particle_hi=np.array([ 2.0,  2.0, 0.25]),
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

    # 解析参数并初始化查看器
    viewer, args = newton.examples.init(parser)

    # 创建示例并运行
    example = Example(viewer, args)
    newton.examples.run(example, args)