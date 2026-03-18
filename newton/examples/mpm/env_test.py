import numpy as np
from types import SimpleNamespace

class NewtonRobotEnv:
    """最小可跑通的环境骨架（不依赖 gym）。

    先用它把训练脚本的 loop / 日志 / 随机数稳定性跑通；
    下一步再把 obs/reward/done 替换成真实仿真（robot_terrain）。
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        max_steps: int = 20,
        seed: int | None = 0,
        action_cost: float = 0.05,
    ):
        """构造一个 toy 环境（不依赖真实仿真）。

        Args:
            state_dim: 观测向量维度。
            action_dim: 动作向量维度。
            max_steps: 单个 episode 的最大步数。
            seed: 随机数种子（None 表示不固定）。
            action_cost: 动作惩罚系数，reward = 1 - action_cost * |a|^2。

        Returns:
            None
        """
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)
        self.action_cost = float(action_cost)
        self.step_count = 0
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: int | None = 0):
        """设置随机数种子。

        Args:
            seed: 新的随机数种子。

        Returns:
            None
        """
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """重置环境并返回初始观测。

        Returns:
            np.ndarray: 初始观测，形状 `(state_dim,)`。
        """
        self.step_count = 0
        return np.zeros(self.state_dim, dtype=np.float32)

    def step(self, action):
        """推进环境一步。

        Args:
            action: 动作向量，形状 `(action_dim,)`。

        Returns:
            obs: np.ndarray，下一观测，形状 `(state_dim,)`。
            reward: float，标量奖励。
            done: bool，episode 是否结束。
            info: dict，额外信息（这里包含 step_count）。
        """
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.action_dim}, got {action.shape[0]}")

        obs = self.rng.standard_normal(self.state_dim, dtype=np.float32)

        # 让 reward 与 action 挂钩：动作幅度越大，奖励越低（便于你观察策略输出影响回报）。
        reward = 1.0 - self.action_cost * float(np.sum(action * action))
        done = self.step_count >= self.max_steps
        info = {"step_count": int(self.step_count)}
        return obs, float(reward), bool(done), info

    def close(self):
        """关闭环境并释放资源（toy env 无资源，空实现）。"""
        return


class RobotTerrainEnv:
    """用 `robot_terrain.py` 的 Example 包装成最小 RL 环境。

    目标：先把“env 能驱动真实仿真 step()”跑通。
    - obs: [fx, fy, fz, |f|] （来自 Example.foot_particle_force_world）
    - action: 直接写入 Example.dmp_action（长度必须匹配）
    - reward: 0.01*|f| - action_cost*|a|^2 （先很粗糙，后续再改）
    """

    def __init__(
        self,
        max_steps: int = 20,
        seed: int | None = 0,
        action_cost: float = 0.001,
        force_reward_scale: float = 0.01,
        fps: float = 160.0,
        substeps: int = 32,
        device: str | None = None,
    ):
        """构造基于 `robot_terrain.Example` 的最小 RL 环境封装。

        Args:
            max_steps: 单个 episode 最大步数。
            seed: 随机种子（目前主要用于上层脚本一致性；Example 内部未必使用）。
            action_cost: 动作惩罚系数（越大越鼓励小动作）。
            force_reward_scale: 接触力幅值奖励系数（reward 里正项）。
            fps: 仿真帧率（传入 Example options）。
            substeps: 每帧子步数（传入 Example options）。
            device: Warp 设备名（如 "cpu"/"cuda:0"），可通过 `WP_DEVICE` 环境变量覆盖。

        Returns:
            None

        观测/动作约定:
            - obs (state_dim=9): `[foot_pos_rel(3), fx, fy, fz, prev_action(3)]`
            - action (action_dim=3): 直接写入 `example.dmp_action`。
        """
        # 参考 legged_gym 的 6D 状态设计：
        #   [foot_pos_rel(3), contact_force(3)]
        # 这里按你的要求保留：fx,fy,fz + prev_action(3)，因此 obs 为：
        #   [foot_pos_rel(3), fx, fy, fz, prev_action(3)]
        self.state_dim = 9
        self.action_dim = 3

        self.max_steps = int(max_steps)
        self.seed_value = seed
        self.action_cost = float(action_cost)
        self.force_reward_scale = float(force_reward_scale)

        self.fps = float(fps)
        self.substeps = int(substeps)
        self.device = device

        self.step_count = 0
        self._example = None
        self._viewer = None
        self._prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        self.reset()

    def seed(self, seed: int | None = 0):
        """设置环境 seed（保存到实例字段，供 reset 时使用）。

        Args:
            seed: 新的随机种子。

        Returns:
            None
        """
        self.seed_value = seed

    def _make_example(self):
        """创建 `robot_terrain.Example` 以及对应 viewer。

        Returns:
            tuple: `(viewer, example)`。

        说明:
            - 默认使用 `ViewerNull`，避免训练时打开窗口。
            - 若设置了 `device` 或 `WP_DEVICE`，会尝试在创建 Warp 对象前切换设备。
        """
        import os
        import warp as wp
        import newton.viewer
        from newton.examples.mpm import robot_terrain

        # 如果用户显式请求 device，则在构建任何 Warp/Newton 对象前切换。
        # 注意：这无法“强制”让 GPU 可用；CUDA 驱动/硬件不可用时仍会回退或报错。
        requested_device = self.device or os.getenv("WP_DEVICE")
        if requested_device:
            try:
                wp.init()
                wp.set_device(requested_device)
            except Exception:
                # 保持最小可运行：device 不可用时继续用默认设备（通常是 cpu）
                pass

        viewer = newton.viewer.ViewerNull(num_frames=self.max_steps)

        # robot_terrain.Example 只强依赖少数 options 字段，其余用 getattr 有默认。
        options = SimpleNamespace(
            fps=self.fps,
            substeps=self.substeps,
            num_frames=self.max_steps,
            viewer="null",
            headless=True,
            test=False,
            # 关掉 matplotlib 交互绘图（env 训练时不需要）
            enable_plot=False,
            # 初始 action（长度为 3：LF_HAA/LF_HFE/LF_KFE）
            dmp_action=[0.0, 0.0, 0.0],
        )

        example = robot_terrain.Example(viewer, options)

        # RL 训练不希望 headless 模式“接触后自动退出进程”。
        # 这里强制关闭该逻辑（若字段不存在则忽略）。
        if hasattr(example, "_headless_null_viewer"):
            try:
                example._headless_null_viewer = False
            except Exception:
                pass
        return viewer, example

    def _get_obs(self) -> np.ndarray:
        """从 Example 中读取当前观测。

        Returns:
            np.ndarray: 观测向量，形状 `(state_dim,)`，按约定拼接：
                `[foot_pos_rel(3), fx, fy, fz, prev_action(3)]`。
        """
        if self._example is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        f = getattr(self._example, "foot_particle_force_world", None)
        if f is None:
            f = np.zeros(3, dtype=np.float32)
        f = np.asarray(f, dtype=np.float32).reshape(3)

        # 位姿：参考实现里常用“足端相对基座位置”(3)。
        foot_pos_rel = np.zeros(3, dtype=np.float32)
        try:
            state = getattr(self._example, "state_0", None)
            foot_id = int(getattr(self._example, "foot", -1))
            base_id = int(getattr(self._example, "base", 0))
            if state is not None and foot_id >= 0:
                body_q = state.body_q.numpy()
                foot_pos = np.asarray(body_q[foot_id][0:3], dtype=np.float32).reshape(3)
                base_pos = np.asarray(body_q[base_id][0:3], dtype=np.float32).reshape(3) if base_id >= 0 else np.zeros(3, dtype=np.float32)
                foot_pos_rel = foot_pos - base_pos
        except Exception:
            foot_pos_rel = np.zeros(3, dtype=np.float32)

        obs = np.concatenate(
            [
                np.asarray(foot_pos_rel, dtype=np.float32).reshape(3),
                np.asarray(f, dtype=np.float32).reshape(3),
                np.asarray(self._prev_action, dtype=np.float32).reshape(self.action_dim),
            ],
            axis=0,
        )
        return obs.astype(np.float32, copy=False)

    def reset(self):
        """重置仿真并返回初始观测。

        Returns:
            np.ndarray: 初始观测，形状 `(state_dim,)`。
        """
        self.close()
        self._viewer, self._example = self._make_example()
        self.step_count = 0
        self._prev_action = np.zeros((self.action_dim,), dtype=np.float32)
        return self._get_obs()

    def step(self, action):
        """将动作注入仿真并推进一步。

        Args:
            action: 动作向量，形状 `(action_dim,)`，会写入 `example.dmp_action`。

        Returns:
            obs: np.ndarray，下一观测，形状 `(state_dim,)`。
            reward: float，标量奖励，当前实现为：
                `force_reward_scale * |f| - action_cost * |a|^2`。
            done: bool，episode 是否结束（达到 max_steps 或仿真内部触发 SystemExit）。
            info: dict，包含 `step_count`、`force_mag`，以及可选的 `terminated_by`。
        """
        if self._example is None:
            raise RuntimeError("Environment not reset()")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.action_dim}, got {action.shape[0]}")

        # 把 RL action 注入到 robot_terrain 的 DMP 外力项
        if hasattr(self._example, "dmp_action"):
            self._example.dmp_action = action.copy()

        # 让下一时刻 obs 可见上一时刻 action
        self._prev_action = action.copy()

        try:
            self._example.step()
        except SystemExit:
            # robot_terrain 里可能会触发 SystemExit（例如 headless 快速验证路径）。
            # 这里把它转成 episode 结束，避免直接退出训练进程。
            self.step_count += 1
            obs = self._get_obs()
            # obs 结构为 [foot_pos_rel(3), fx,fy,fz, prev_action(3)]
            force_vec = np.asarray(obs[3:6], dtype=np.float32) if obs.shape[0] >= 6 else np.zeros(3, dtype=np.float32)
            force_mag = float(np.linalg.norm(force_vec))
            reward = self.force_reward_scale * force_mag - self.action_cost * float(np.sum(action * action))
            info = {"step_count": int(self.step_count), "force_mag": force_mag, "terminated_by": "SystemExit"}
            return obs, float(reward), True, info
        self.step_count += 1

        obs = self._get_obs()
        force_vec = np.asarray(obs[3:6], dtype=np.float32)
        force_mag = float(np.linalg.norm(force_vec))

        reward = self.force_reward_scale * force_mag - self.action_cost * float(np.sum(action * action))
        done = self.step_count >= self.max_steps
        info = {"step_count": int(self.step_count), "force_mag": force_mag}
        return obs, float(reward), bool(done), info

    def close(self):
        """关闭 viewer/example，释放资源。"""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
        self._viewer = None
        self._example = None