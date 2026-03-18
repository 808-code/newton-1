from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


@dataclass
class OnPolicyBatch:
    """一段 on-policy rollout 的 numpy 批数据。

    字段说明:
        state: (N, state_dim)
        action: (N, action_dim)
        reward: (N,)
        not_done: (N,) 其中 1.0 表示未终止，0.0 表示终止
        next_state: (N, state_dim)
        old_logprob: 可选 (N,) 采样时的 log pi(a|s)
    """

    state: "np.ndarray"
    action: "np.ndarray"
    reward: "np.ndarray"
    not_done: "np.ndarray"
    next_state: "np.ndarray"
    old_logprob: Optional["np.ndarray"]


class OnPolicyRolloutBuffer:
    """最小 on-policy 轨迹缓冲区。

    只负责采样收集：存 (s, next_s, a, r, done[, logp])。
    训练时用 `on_policy_sample()` 一次性取出整段 rollout 给 `PPO.train()`。
    """

    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        """初始化缓冲区并预分配内存。

        Args:
            state_dim: 状态维度。
            action_dim: 动作维度。
            capacity: 最大可存的时间步数量 N。

        Returns:
            None
        """
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._not_done = np.zeros((self.capacity,), dtype=np.float32)
        self._logprobs = np.zeros((self.capacity,), dtype=np.float32)
        self._has_logprob = False

        self._size = 0

    def reset(self) -> None:
        """清空当前已写入的数据（不重新分配内存）。

        Returns:
            None

        用法:
            通常每个 episode/rollout 开始前调用一次。
        """
        self._size = 0
        self._has_logprob = False

    def __len__(self) -> int:
        """返回当前已存的转移条数。"""
        return int(self._size)

    def add(
        self,
        state: ArrayLike,
        next_state: ArrayLike,
        action: ArrayLike,
        reward: float,
        done: bool,
        logprob: Optional[float] = None,
    ) -> None:
        """写入一条转移样本。

        Args:
            state: 状态（可为 list/tuple/np.ndarray），形状 `(state_dim,)`。
            next_state: 下一状态，形状 `(state_dim,)`。
            action: 动作，形状 `(action_dim,)`。
            reward: 标量奖励。
            done: 是否终止（True 表示终止/截断）。
            logprob: 可选，采样时的 $\log \pi(a|s)$（标量）。

        Returns:
            None

        说明:
            - 内部会做维度校验，不匹配会抛 `ValueError`。
            - `not_done` 以 1.0/0.0 存储，方便后续 bootstrap。
            - 如果从未传入 `logprob`，则 `as_numpy().old_logprob` 为 None。
        """
        if self._size >= self.capacity:
            raise RuntimeError(f"buffer overflow: size={self._size} capacity={self.capacity}")

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        ns = np.asarray(next_state, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if s.shape[0] != self.state_dim:
            raise ValueError(f"state_dim mismatch: expected {self.state_dim}, got {s.shape[0]}")
        if ns.shape[0] != self.state_dim:
            raise ValueError(f"state_dim mismatch: expected {self.state_dim}, got {ns.shape[0]}")
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action_dim mismatch: expected {self.action_dim}, got {a.shape[0]}")

        idx = self._size
        self._states[idx] = s
        self._next_states[idx] = ns
        self._actions[idx] = a
        self._rewards[idx] = float(reward)
        self._not_done[idx] = 0.0 if bool(done) else 1.0

        if logprob is not None:
            self._logprobs[idx] = float(logprob)
            self._has_logprob = True

        self._size += 1

    def as_numpy(self) -> OnPolicyBatch:
        """导出当前 rollout 为 numpy 批数据（会 copy）。

        Returns:
            OnPolicyBatch: 仅包含 `[0:self._size]` 的有效部分。

        说明:
            返回的是 copy，避免训练时意外修改到 buffer 内部数组。
        """
        old_logprob = self._logprobs[: self._size].copy() if self._has_logprob else None
        return OnPolicyBatch(
            state=self._states[: self._size].copy(),
            next_state=self._next_states[: self._size].copy(),
            action=self._actions[: self._size].copy(),
            reward=self._rewards[: self._size].copy(),
            not_done=self._not_done[: self._size].copy(),
            old_logprob=old_logprob,
        )

    def on_policy_sample(self):
        """返回 `PPO.train()` 期望的 torch batch tuple。

        Returns:
            若 buffer 中写入过 `logprob`：
                6 元组 `(state, action, reward, not_done, next_state, old_logprob)`
            否则：
                5 元组 `(state, action, reward, not_done, next_state)`

        形状约定:
            - state: `(N, state_dim)`
            - action: `(N, action_dim)`
            - reward: `(N,)`
            - not_done: `(N,)`
            - next_state: `(N, state_dim)`
            - old_logprob: `(N,)`

        注意:
            这里不指定 device；`PPO.train()` 内部会把张量搬到其 device。
        """
        import torch

        batch = self.as_numpy()
        state_t = torch.as_tensor(batch.state, dtype=torch.float32)
        next_state_t = torch.as_tensor(batch.next_state, dtype=torch.float32)
        action_t = torch.as_tensor(batch.action, dtype=torch.float32)
        reward_t = torch.as_tensor(batch.reward, dtype=torch.float32)
        not_done_t = torch.as_tensor(batch.not_done, dtype=torch.float32)

        if batch.old_logprob is None:
            return state_t, action_t, reward_t, not_done_t, next_state_t

        old_logprob_t = torch.as_tensor(batch.old_logprob, dtype=torch.float32)
        return state_t, action_t, reward_t, not_done_t, next_state_t, old_logprob_t
