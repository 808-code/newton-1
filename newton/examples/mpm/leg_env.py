from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

import newton
import newton.examples
from newton.examples.mpm.terrain_leg import Example as TerrainLegExample


class NullViewer:
    """无头 (headless) 训练用的空 Viewer。

    目的：复用 Newton example 的仿真逻辑，但在强化学习训练时通常不需要渲染。
    `TerrainLegExample` 内部会调用 Viewer 的若干接口（set_model/log_points/...），
    这里提供同名方法但不做任何事，以避免引入窗口/GUI 依赖。
    """

    show_particles: bool = False

    def set_model(self, _model: Any) -> None:
        return

    def register_ui_callback(self, _fn: Any, position: str = "side") -> None:
        return

    def apply_forces(self, _state: Any) -> None:
        return

    def begin_frame(self, _t: float) -> None:
        return

    def log_state(self, _state: Any) -> None:
        return

    def log_points(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def log_lines(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def end_frame(self) -> None:
        return


def _build_default_options() -> argparse.Namespace:
    """构造一个与 `terrain_leg.py` 兼容的 options（argparse.Namespace）。

    注意：这里刻意不调用 `newton.examples.init()`。
    - `init()` 通常会创建 Viewer、解析命令行、配置可视化等。
    - 强化学习训练一般是 headless（无窗口），因此用手工构造参数最稳妥。
    """

    parser = newton.examples.create_parser()

    # Scene
    parser.add_argument("--collider", default="none", choices=["cube", "none"], type=str)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-0.5, -0.5, 0.0])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[0.5, 0.5, 0.25])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=160)
    parser.add_argument("--substeps", type=int, default=32)

    # Material
    parser.add_argument("--density", type=float, default=2500.0)
    parser.add_argument("--air-drag", type=float, default=1.0)
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0)
    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e15)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.48)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e5)
    parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--hardening", type=float, default=0.0)

    # MPM
    parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
    parser.add_argument("--solver", "-s", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"])
    parser.add_argument("--transfer-scheme", "-ts", type=str, default="pic", choices=["apic", "pic"])
    parser.add_argument("--strain-basis", "-sb", type=str, default="P0", choices=["P0", "Q1"])
    parser.add_argument("--max-iterations", "-it", type=int, default=50)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)

    # DMP control (2D)
    parser.add_argument("--dmp-action", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--dmp-force-scale", type=float, default=500.0)
    parser.add_argument("--dmp-fn-max", type=float, default=400.0)
    parser.add_argument("--dmp-ft-max", type=float, default=400.0)
    parser.add_argument("--joint-target-ke", type=float, default=2000.0)
    parser.add_argument("--joint-target-kd", type=float, default=50.0)

    # RL/headless specific
    parser.add_argument("--enable-plot", action="store_true", default=False)

    return parser.parse_args([])


@dataclass
class LegEnvConfig:
    """环境侧的轻量配置。

    这些配置不属于物理引擎参数（那些在 options 里），而是训练接口相关：
    - max_episode_steps：每个 episode 的最大步数。
    - force_scale_obs：观测里力的归一化因子（缩放到更适合网络的量级）。
    - action_clip：动作裁剪范围（最终动作会进入 DMP 外力项）。
    """

    max_episode_steps: int = 200
    force_scale_obs: float = 10.0
    action_clip: float = 1.0


class LegEnv:
    """无头 Newton leg+MPM 强化学习环境封装。

        设计目标：
        - 复用 `newton/examples/mpm/terrain_leg.py` 的仿真（两关节腿 + MPM 地形 + DMP 耦合）。
        - 提供一个最小 Gym-like 接口：`reset()` / `step(action)` -> (obs, reward, done, info)。

        说明（本环境的定义）：
        - 状态 obs（6维）：
            1) q[0:2]   : 关节角（hip/knee，rad）
            2) qd[0:2]  : 关节角速度（rad/s）
            3) [F_n, F_t] / force_scale_obs : 足端受力（法向/切向）做缩放
        - 动作 action（2维，范围 [-1,1]）：
            对应 `TerrainLegExample.dmp_action`，用于 DMP 外力项 `force_eta = action * dmp_force_scale`。
    """

    def __init__(self, config: Optional[LegEnvConfig] = None, options: Optional[argparse.Namespace] = None):
        self.config = config or LegEnvConfig()
        self._base_options = options or _build_default_options()
        self._viewer = NullViewer()

        # ========== 强化学习接口维度定义 ==========
        # state_dim/action_dim 会被 PPO 使用（建网络、检查 action shape 等）。
        self.state_dim = 6
        self.action_dim = 2

        self._episode_steps = 0
        self._sim: Optional[TerrainLegExample] = None

        self.reset()

    def _get_obs(self) -> np.ndarray:
        """从仿真状态拼出观测向量 obs。

        未知量/符号说明：
        - q, qd: Newton articulation 的关节广义坐标及速度（这里只取前2个 DOF）。
        - body_sand_forces: MPM->刚体耦合得到的空间力（6维），这里取线性力 [:3]。
        - f_n: 法向力，约定为世界坐标 z 方向的“向上”部分 max(0, fz)。
        - f_t: 切向力，约定为世界坐标 x/y 平面合力 sqrt(fx^2 + fy^2)。
        """
        assert self._sim is not None
        q = self._sim.state_0.joint_q.numpy().astype(np.float32)
        qd = self._sim.state_0.joint_qd.numpy().astype(np.float32)

        f = self._sim.body_sand_forces.numpy()[self._sim.foot][:3].astype(np.float32)
        fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
        f_n = max(0.0, fz)
        f_t = float(np.sqrt(fx * fx + fy * fy))

        obs = np.concatenate(
            [q[:2], qd[:2], np.array([f_n, f_t], dtype=np.float32) / float(self.config.force_scale_obs)],
            dtype=np.float32,
        )
        return obs

    def reset(self) -> np.ndarray:
        """重置环境并返回初始观测。

        当前实现：每个 episode 重新 new 一个 `TerrainLegExample`。
        - 好处：简单、状态完全干净（MPM 粒子/网格/接触缓存全部重置）。
        - 代价：初始化较慢；如果后续想提速，可做“真 reset”（复位 state 而不重建）。
        """
        self._episode_steps = 0

        # Re-create simulator each episode for a clean state.
        # Note: for speed, this can later be replaced by a true in-sim reset.
        opts = argparse.Namespace(**vars(self._base_options))
        opts.enable_plot = False

        self._sim = TerrainLegExample(self._viewer, opts)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """推进环境一步。

        Args:
            action: 形状 (2,) 的连续动作，范围建议 [-1,1]。

        Returns:
            obs: 下一时刻观测 (6,)
            reward: 标量奖励
            done: episode 是否结束
            info: 额外信息（调试用，不参与训练也可）
        """
        assert self._sim is not None
        self._episode_steps += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")

        action = np.clip(action, -self.config.action_clip, self.config.action_clip)
        # 关键：把动作写到仿真的 DMP 输入上。
        # 在 `terrain_leg.py` 内部，每个子步会用：force_eta = clip(dmp_action) * dmp_force_scale。
        self._sim.dmp_action = action

        self._sim.step()

        obs = self._get_obs()

        # ========== 奖励函数（可按任务目标替换） ==========
        # 这里给一个最保守的默认：惩罚接触力，鼓励“轻触/不猛踩”。
        # 注意 obs 里力被除以 force_scale_obs，这里再乘回来得到真实量级。
        f_n, f_t = float(obs[-2] * self.config.force_scale_obs), float(obs[-1] * self.config.force_scale_obs)
        reward = -(f_n + f_t) * 0.01

        done = False
        if self._episode_steps >= self.config.max_episode_steps:
            done = True

        info = {
            "f_n": f_n,
            "f_t": f_t,
            "step": self._episode_steps,
        }
        return obs, float(reward), done, info

    def render(self) -> None:
        # 训练时通常无需渲染；保留接口以兼容一些训练脚本。
        return
