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

    `TerrainLegExample` 会调用 viewer 的若干方法；这里提供同名空实现，避免打开窗口。
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
    """构造一个与 `terrain_leg.py` 兼容的 options（argparse.Namespace）。"""
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

    # DMP control (terrain_leg.py 内部是 2D；env1 提供 3D，其中第3维用作 force_scale 调节)
    parser.add_argument("--dmp-action", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--dmp-force-scale", type=float, default=500.0)
    parser.add_argument("--dmp-fn-max", type=float, default=400.0)
    parser.add_argument("--dmp-ft-max", type=float, default=400.0)
    parser.add_argument("--joint-target-ke", type=float, default=2000.0)
    parser.add_argument("--joint-target-kd", type=float, default=50.0)

    # Headless
    parser.add_argument("--enable-plot", action="store_true", default=False)

    return parser.parse_args([])


@dataclass
class LeggedRobotConfig:
    """训练接口相关配置（不涉及底层物理引擎参数）。"""

    max_episode_steps: int = 200
    force_scale_obs: float = 10.0
    action_clip: float = 1.0
    # 第3维动作用于调节 dmp_force_scale：mult = clamp(1 + k*a3, [min,max])
    dmp_force_scale_mult_k: float = 0.5
    dmp_force_scale_mult_min: float = 0.1
    dmp_force_scale_mult_max: float = 2.0


class LeggedRobot:
    """Newton 单腿+MPM 地形环境（env1：可用版）。

    这是对 `terrain_leg.py` 的最小封装，提供 PPO 常用的 `reset/step` 接口。

    观测 (state_dim=6)：
    - q[0:2], qd[0:2], [F_n, F_t]/force_scale_obs

    动作 (action_dim=3)：
    - a[0:2]：写入 `TerrainLegExample.dmp_action`（2D 外力项）
    - a[2]：调节 `dmp_force_scale` 的倍率（用于把 2D DMP 控制扩展成 3D 动作空间）
    """

    def __init__(self, config: Optional[LeggedRobotConfig] = None, options: Optional[argparse.Namespace] = None):
        self.config = config or LeggedRobotConfig()
        self._base_options = options or _build_default_options()
        self._viewer = NullViewer()

        self.state_dim = 6
        self.action_dim = 3

        self._episode_steps = 0
        self._sim: Optional[TerrainLegExample] = None
        self._base_dmp_force_scale: float = float(getattr(self._base_options, "dmp_force_scale", 500.0))

        self.reset()

    def _get_obs(self) -> np.ndarray:
        assert self._sim is not None
        q = self._sim.state_0.joint_q.numpy().astype(np.float32)
        qd = self._sim.state_0.joint_qd.numpy().astype(np.float32)

        f = self._sim.body_sand_forces.numpy()[self._sim.foot][:3].astype(np.float32)
        fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
        f_n = max(0.0, fz)
        f_t = float(np.sqrt(fx * fx + fy * fy))

        return np.concatenate(
            [q[:2], qd[:2], np.array([f_n, f_t], dtype=np.float32) / float(self.config.force_scale_obs)],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self._episode_steps = 0

        opts = argparse.Namespace(**vars(self._base_options))
        opts.enable_plot = False
        self._sim = TerrainLegExample(self._viewer, opts)
        # 记录实例化后的基准值（避免 options 被外部改动）
        self._base_dmp_force_scale = float(getattr(self._sim, "dmp_force_scale", self._base_dmp_force_scale))
        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self._sim is not None
        self._episode_steps += 1

        a = np.asarray(actions, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"Expected action shape ({self.action_dim},), got {a.shape}")
        a = np.clip(a, -self.config.action_clip, self.config.action_clip)

        # 1) 前两维：DMP 外力项（terrain_leg 内部会乘 dmp_force_scale）
        self._sim.dmp_action = a[:2].copy()

        # 2) 第三维：调节 dmp_force_scale 的倍率，作为“第3维动作”
        mult = 1.0 + float(self.config.dmp_force_scale_mult_k) * float(a[2])
        mult = float(np.clip(mult, self.config.dmp_force_scale_mult_min, self.config.dmp_force_scale_mult_max))
        self._sim.dmp_force_scale = float(self._base_dmp_force_scale) * mult

        self._sim.step()
        obs = self._get_obs()

        f_n = float(obs[-2] * self.config.force_scale_obs)
        f_t = float(obs[-1] * self.config.force_scale_obs)
        reward = -(f_n + f_t) * 0.01

        done = self._episode_steps >= self.config.max_episode_steps
        info = {"f_n": f_n, "f_t": f_t, "step": self._episode_steps, "dmp_force_scale_mult": mult}
        return obs, float(reward), bool(done), info

    def render(self) -> None:
        return


if __name__ == "__main__":
    env = LeggedRobot()
    s = env.reset()
    for _ in range(5):
        s, r, d, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
        print("step", info["step"], "r", r, "done", d, "f_n", info["f_n"], "f_t", info["f_t"], "mult", info["dmp_force_scale_mult"])

