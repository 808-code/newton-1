import os

from .env_test import NewtonRobotEnv, RobotTerrainEnv
from .on_policy_buffer import OnPolicyRolloutBuffer
import numpy as np


def _maybe_save_histogram(force_mags: np.ndarray, rewards: np.ndarray, out_path: str) -> bool:
    """尝试保存随机动作的统计直方图（如果 matplotlib 可用）。

    Args:
        force_mags: 接触力幅值数组，形状 `(N,)`。
        rewards: reward 数组，形状 `(N,)`。
        out_path: 输出图片路径（如 `sanity_check_hist.png`）。

    Returns:
        bool: 保存成功返回 True；如果缺少 matplotlib 等依赖则返回 False。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(force_mags, bins=50)
    axes[0].set_title("|f| histogram")
    axes[0].set_xlabel("|f|")
    axes[0].set_ylabel("count")

    axes[1].hist(rewards, bins=50)
    axes[1].set_title("reward histogram")
    axes[1].set_xlabel("reward")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def sanity_check(env, episodes: int, seed: int, max_steps: int):
    """用随机动作跑若干个 episode，检查 reward/力是否“非零且有方差”。

    这一步的目的是快速验证：
    - 环境 step() 真在变化（不是恒定 0）
    - reward 与观测里的力项大致合理（均值/方差不为 0）

    Args:
        env: 环境对象，需要实现 `reset()`/`step(action)`，并有 `action_dim` 字段。
        episodes: episode 数。
        seed: 随机种子（用于采样随机动作）。
        max_steps: 每个 episode 的最大步数。

    Returns:
        None

    输出:
        - 打印 mean/std/min/max/near0_frac 等统计
        - 若设置 `SANITY_PLOT` 或默认路径，则输出直方图图片
    """
    rng = np.random.default_rng(int(seed))

    all_force = []
    all_reward = []

    for _ep in range(int(episodes)):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < int(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
            next_obs, reward, done, info = env.step(action)

            # 尽量从 info 里拿（robot env 有），否则从 obs[3]（我们约定了 |f| 在 obs[3]）。
            if isinstance(info, dict) and "force_mag" in info:
                force_mag = float(info["force_mag"])
            else:
                obs_vec = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                # RobotTerrainEnv: [foot_pos_rel(3), fx,fy,fz, prev_action(3)]
                if obs_vec.shape[0] >= 6:
                    force_mag = float(np.linalg.norm(obs_vec[3:6]))
                else:
                    force_mag = 0.0

            all_force.append(force_mag)
            all_reward.append(float(reward))

            obs = next_obs
            steps += 1

    force_mags = np.asarray(all_force, dtype=np.float64)
    rewards = np.asarray(all_reward, dtype=np.float64)

    def _stats(x: np.ndarray):
        return float(np.mean(x)), float(np.std(x))

    f_mean, f_std = _stats(force_mags) if force_mags.size else (0.0, 0.0)
    r_mean, r_std = _stats(rewards) if rewards.size else (0.0, 0.0)

    near0_f = float(np.mean(np.abs(force_mags) < 1e-12)) if force_mags.size else 1.0
    near0_r = float(np.mean(np.abs(rewards) < 1e-12)) if rewards.size else 1.0

    print("=== SANITY CHECK (random actions) ===")
    print(f"episodes={int(episodes)}  max_steps={int(max_steps)}  samples={int(rewards.size)}")
    print(f"mean(|f|)={f_mean:.6g}  std(|f|)={f_std:.6g}  near0_frac={near0_f:.3f}  min={float(np.min(force_mags)) if force_mags.size else 0.0:.6g}  max={float(np.max(force_mags)) if force_mags.size else 0.0:.6g}")
    print(f"mean(reward)={r_mean:.6g}  std(reward)={r_std:.6g}  near0_frac={near0_r:.3f}  min={float(np.min(rewards)) if rewards.size else 0.0:.6g}  max={float(np.max(rewards)) if rewards.size else 0.0:.6g}")

    out_path = os.getenv("SANITY_PLOT", "sanity_check_hist.png")
    if _maybe_save_histogram(force_mags, rewards, out_path):
        print(f"saved_hist={out_path}")
    else:
        print("saved_hist=FAILED (matplotlib not available)")

def main(episodes: int = 5, seed: int = 0, use_robot_terrain: bool = False, max_steps: int = 20):
    """最小 PPO 训练/验证入口。

    Args:
        episodes: 运行多少个 episode。
        seed: 随机种子。
        use_robot_terrain: True 使用 `RobotTerrainEnv`（真实仿真）；False 使用 `NewtonRobotEnv`（toy）。
        max_steps: 每个 episode 的最大步数。

    Returns:
        None

    环境变量开关:
        - `USE_ROBOT_TERRAIN=1`：使用真实仿真 env
        - `SANITY_CHECK=1`：先跑随机动作统计并退出（不训练）
        - `DO_TRAIN=1`：每个 episode 结束后执行一次 `agent.train(buffer)`
        - `EPISODES`/`MAX_STEPS`/`SEED`：覆盖默认参数
    """
    from .PPO import PPO  # 延迟导入：避免在 import 时触发 torch/CUDA 初始化
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    if use_robot_terrain:
        env = RobotTerrainEnv(max_steps=max_steps, seed=seed)
    else:
        env = NewtonRobotEnv(max_steps=max_steps, seed=seed)
    env.seed(seed)

    # 优先做 reward 可学习性检查：用随机动作跑多集，统计 |f| 和 reward 的均值/方差。
    if os.getenv("SANITY_CHECK", "0") in ("1", "true", "True"):
        sanity_check(env, episodes=int(episodes), seed=int(seed), max_steps=int(max_steps))
        env.close()
        return

    agent = PPO(state_dim=env.state_dim, action_dim=env.action_dim, max_action=1.0, hidden_dim=64)

    returns = []
    for ep in range(int(episodes)):
        obs = env.reset()
        buffer = OnPolicyRolloutBuffer(state_dim=env.state_dim, action_dim=env.action_dim, capacity=max_steps)
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_action_l2_sq = 0.0

        while not done:
            action, logprob = agent.select_action_with_logprob(obs, deterministic=False)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape[0] != env.action_dim:
                action = action.reshape(env.action_dim)

            ep_action_l2_sq += float(np.sum(action * action))

            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, next_obs, action, reward, done, logprob=float(np.asarray(logprob).reshape(-1)[0]))
            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1

        returns.append(ep_ret)

        # 默认先只收集数据；需要训练时再打开开关。
        if os.getenv("DO_TRAIN", "0") in ("1", "true", "True"):
            agent.train(buffer)

        mean_l2_sq = ep_action_l2_sq / max(ep_len, 1)
        print(
            f"ep={ep:03d}  return={ep_ret:.2f}  len={ep_len}  mean|a|^2={mean_l2_sq:.3f}  "
            f"buffer={len(buffer)}  info={info}"
        )

    print(f"avg_return={float(np.mean(returns)):.2f}  episodes={len(returns)}")
    env.close()


if __name__ == "__main__":
    # 保持最简单：先把闭环跑通。后续我们再加 buffer + PPO.train()
    use_robot_terrain = os.getenv("USE_ROBOT_TERRAIN", "0") in ("1", "true", "True")
    episodes = int(os.getenv("EPISODES", "5"))
    max_steps = int(os.getenv("MAX_STEPS", "20"))
    seed = int(os.getenv("SEED", "0"))
    main(episodes=episodes, seed=seed, use_robot_terrain=use_robot_terrain, max_steps=max_steps)