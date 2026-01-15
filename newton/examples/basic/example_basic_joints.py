# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Joints
#
# Shows how to use the ModelBuilder API to programmatically create different
# joint types: BALL, DISTANCE, PRISMATIC, and REVOLUTE.
#
# Command: python -m newton.examples basic_joints
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # ========== 仿真时间控制 ==========
        # 与 terrain_leg 相同：先定义 fps / frame_dt / substeps / sim_dt，后续所有 step 都依赖这些量。
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # ========== Viewer 与命令行参数 ==========
        self.viewer = viewer
        self.args = args

        # ========== 刚体模型构建：ModelBuilder ==========
        builder = newton.ModelBuilder()

        # 添加地面（静态碰撞体）
        builder.add_ground_plane()

        # ========== 几何尺寸（全局复用） ==========
        cuboid_hx = 0.1
        cuboid_hy = 0.1
        cuboid_hz = 0.75
        upper_hz = 0.25 * cuboid_hz

        # ========== 场景布局参数 ==========
        # 将不同 joint demo 摆成三排（沿 y 轴分开），从高处落下便于观察约束。
        rows = [-3.0, 0.0, 3.0]
        drop_z = 2.0

        # ========== REVOLUTE（转动 / 铰链）关节演示 ==========
        # 目标：构造一个“世界 -> 固定锚点 -> 铰链连杆”的两关节链。
        # - j_fixed_rev：把 a_rev 固定到世界（parent=-1 表示世界坐标系）。
        # - j_revolute：把 b_rev 通过转轴约束连接到 a_rev，实现单自由度转动。
        # 关键点：joint 的 parent_xform/child_xform 定义的是“在父/子局部坐标系下”的关节安装位姿。
        y = rows[0]

        # a_rev：上方锚点（之后会被 fixed 到世界），先给一个世界初始位姿
        a_rev = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
        # b_rev：下方可动连杆，初始姿态略微旋转，避免完全对称导致观察不明显
        b_rev = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15)
            ),
            key="b_rev",
        )

        # 几何：给 link 挂 shape，shape 会参与渲染与碰撞（默认是刚体）
        builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # 关节 1：固定关节，把 a_rev 锚定到世界
        # - parent=-1 表示“世界”
        # - parent_xform：关节在世界中的安装位姿（这里选 a_rev 的初始位姿）
        # - child_xform：关节在子体(a_rev)局部中的安装位姿（这里为零位姿）
        j_fixed_rev = builder.add_joint_fixed(
            parent=-1,
            child=a_rev,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_revolute_anchor",
        )

        # 关节 2：转动关节，把 b_rev 铰接到 a_rev
        # - axis：关节转轴（在父坐标系下定义），此处为 X 轴
        # - parent_xform：关节在父体(a_rev)局部的位置：放在 a_rev 的底端（-upper_hz）
        # - child_xform：关节在子体(b_rev)局部的位置：放在 b_rev 的顶端（+cuboid_hz）
        j_revolute = builder.add_joint_revolute(
            parent=a_rev,
            child=b_rev,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="revolute_a_b",
        )

        # articulation：把同一条关节链中的 joint 索引打包
        # 作用：让求解器把它们当作一个整体来解算（约束传播更稳定，也便于后续按 key 查找）。
        builder.add_articulation([j_fixed_rev, j_revolute], key="revolute_articulation")

        # 设置初始转角（单位 rad）：让 b_rev 一开始就处于非零角度
        # 注意：此处直接写 builder.joint_q，对应的是“刚刚添加的那个 revolute 关节”的自由度。
        builder.joint_q[-1] = wp.pi * 0.5

        # ========== PRISMATIC（平移 / 滑动）关节演示 ==========
        # 目标：构造一个“世界 -> 固定锚点 -> 滑块连杆”的两关节链。
        # - j_fixed_pri：把 a_pri 固定到世界
        # - j_prismatic：让 b_pri 只能沿 axis 做直线运动，并设置位移上下限
        # 通过 prismatic 约束，使盒子只能沿给定轴（这里是 Z 轴）做线性滑动。
        y = rows[1]

        # a_pri：上方锚点（固定到世界）
        a_pri = builder.add_link(xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()))
        # b_pri：滑块连杆，初始有轻微姿态扰动
        b_pri = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.12),
            ),
            key="b_prismatic",
        )

        # 几何：两段都是盒子，便于观察滑动与碰撞
        builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # 关节 1：固定关节，把 a_pri 锚定到世界
        j_fixed_pri = builder.add_joint_fixed(
            parent=-1,
            child=a_pri,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_prismatic_anchor",
        )

        # 关节 2：平移关节，连接 a_pri 与 b_pri
        # - axis：允许运动的方向（父坐标系下），此处选择 Z 轴
        # - limit_lower/limit_upper：位移范围（单位：米）
        # - parent_xform/child_xform：与 revolute 类似，用于把关节装到两段几何的端点
        j_prismatic = builder.add_joint_prismatic(
            parent=a_pri,
            child=b_pri,
            axis=wp.vec3(0.0, 0.0, 1.0),  # slide along Z
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            limit_lower=-0.3,
            limit_upper=0.3,
            key="prismatic_a_b",
        )

        # articulation：同样把（固定 + 可动）关节打包
        # articulation 需要 joint 索引列表（这是 Newton 新 API 的要求）
        builder.add_articulation([j_fixed_pri, j_prismatic], key="prismatic_articulation")

        # ========== BALL（球铰）关节演示 ==========
        # 父体用“密度=0”的球体作为运动学锚点（相当于不受动力学影响），子体为可动盒子。
        y = rows[2]
        radius = 0.3
        z_offset = -1.0  # Shift down by 2 units

        # kinematic (massless) sphere as the parent anchor
        a_ball = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity())
        )
        b_ball = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + z_offset), q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.1)
            ),
            key="b_ball",
        )

        rigid_cfg = newton.ModelBuilder.ShapeConfig()
        rigid_cfg.density = 0.0
        builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
        builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        # Connect parent to world
        j_fixed_ball = builder.add_joint_fixed(
            parent=-1,
            child=a_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_ball_anchor",
        )
        j_ball = builder.add_joint_ball(
            parent=a_ball,
            child=b_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="ball_a_b",
        )

        # articulation：将固定到世界的 joint 与 ball joint 组成一个关节链
        builder.add_articulation([j_fixed_ball, j_ball], key="ball_articulation")

        # 设置球铰初始姿态（四元数：roll/pitch/yaw）
        builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)

        # ========== 模型最终化（从 builder -> model） ==========
        self.model = builder.finalize()

        # ========== 求解器 / 状态 / 控制 ==========
        self.solver = newton.solvers.SolverXPBD(self.model)

        # 双缓冲状态：step() 中每个子步都会交换 state_0/state_1
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # 控制对象：用于关节 target / 外力等输入（本例不额外写控制，但保持与其它例子一致）
        self.control = self.model.control()

        # ========== 正向运动学（FK） ==========
        # 对 XPBD 等求解器：在首次 collide / render 之前做一次 FK，确保 body_q / shape 变换已就绪。
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ========== 碰撞检测：Pipeline + Contacts ==========
        # 从命令行参数创建碰撞 pipeline，然后基于当前 state 生成 contacts。
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # ========== Viewer 绑定 ==========
        self.viewer.set_model(self.model)

        # ========== CUDA Graph Capture（可选加速） ==========
        self.capture()

    def capture(self):
        # 与 terrain_leg 类似：CUDA 下可 capture 子步图以减少 launch 开销。
        # 注意：capture 期间不要做 host<->device 数据交换（会导致 capture 失败）。
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # ========== 每帧内部：多子步积分 ==========
        # 每个子步：清力 -> 应用 viewer 交互力 -> 碰撞检测 -> XPBD step -> 交换双缓冲。
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # ========== 单帧推进入口 ==========
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_post_step(self):
        # ========== 运行时断言（每帧） ==========
        # 这些测试用于示例回归：确保关节运动满足预期（转动在平面、滑动在轴、球铰在球面）。
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "revolute motion in plane",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_bottom(qd), wp.vec3(1.0, 0.0, 0.0)))) < 1e-5,
            indices=[self.model.body_key.index("b_rev")],
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "linear motion on axis",
            lambda q, qd: wp.length(abs(wp.cross(wp.spatial_top(qd), wp.vec3(0.0, 0.0, 1.0)))) < 1e-5
            and wp.length(wp.spatial_bottom(qd)) < 1e-5,
            indices=[self.model.body_key.index("b_prismatic")],
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "ball motion on sphere",
            lambda q, qd: abs(wp.dot(wp.spatial_bottom(qd), wp.vec3(0.0, 0.0, 1.0))) < 1e-3,
            indices=[self.model.body_key.index("b_ball")],
        )

    def test_final(self):
        # ========== 结束断言（收尾状态） ==========
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "static bodies are not moving",
            lambda q, qd: max(abs(qd)) == 0.0,
            indices=[2, 4],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "fixed link body has come to a rest",
            lambda q, qd: max(abs(qd)) < 1e-2,
            indices=[0],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "slider link body has come to a rest",
            lambda q, qd: max(abs(qd)) < 1e-5,
            indices=[3],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "movable links are not moving too fast",
            lambda q, qd: max(abs(qd)) < 3.0,
            indices=[1, 5],
        )

    def render(self):
        # ========== 渲染 ==========
        # 记录刚体状态与 contacts，交由 viewer 可视化。
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
