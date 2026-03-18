import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 根据是否有CUDA选择运行设备:有GPU则用GPU,否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Actor-Critic合一网络(非RNN版本)
    - 输入: state(状态)
    - 输出:
      * actor: 动作均值(连续动作, tanh后再乘max_action)
      * critic: 状态价值 V(s)
    """
    def __init__(self, state_dim, action_dim, hidden_dim, max_action, policy_noise=0.2):
        """初始化 Actor-Critic 网络。

        Args:
            state_dim: 状态维度（输入向量长度）。
            action_dim: 动作维度（连续动作向量长度）。
            hidden_dim: 隐藏层宽度。
            max_action: 动作幅值上限（tanh 输出再乘这个系数）。
            policy_noise: 初始策略标准差（用于初始化可学习的 log_std）。

        用法:
            一般由 `PPO` 内部创建，不需要直接调用。
        """
        super(ActorCritic, self).__init__()
        # 非循环分支
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor头
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)

        # 动作缩放上限
        self.max_action = max_action

        # 可学习的 log_std(每个动作维一个)，初始化为 log(policy_noise)
        init_std = max(float(policy_noise), 1e-6)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(np.log(init_std))))

    def forward(self, state):
        """前向特征提取（共享 trunk）。

        Args:
            state: Tensor，形状通常为 `(B, state_dim)`。

        Returns:
            Tensor: 提取后的特征，形状为 `(B, hidden_dim)`。
        """
        p = torch.tanh(self.l1(state))
        p = torch.tanh(self.l2(p))
        return p

    def act(self, state, deterministic=True):
        """根据当前策略输出动作（不返回 logprob）。

        Args:
            state: Tensor，形状通常为 `(B, state_dim)`。
            deterministic: 是否确定性输出。
                - True: 输出动作均值（无采样，适合评估）。
                - False: 从高斯分布采样（适合探索）。

        Returns:
            Tensor: 动作张量，形状 `(B, action_dim)`。

        说明:
            本项目训练时通常使用 `PPO.select_action_with_logprob()`，因为 PPO 需要 old_logprob。
        """
        p = self.forward(state)
        action_mean = torch.tanh(self.actor(p)) * self.max_action

        if deterministic:
            return action_mean

        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        action = dist.sample()
        return action

    def evaluate(self, state, action):
        """评估给定 (state, action) 在当前策略下的值函数和 logprob。

        Args:
            state: Tensor，形状 `(B, state_dim)`。
            action: Tensor，形状 `(B, action_dim)`。

        Returns:
            values: Tensor，形状 `(B,)`，表示 $V(s)$。
            action_logprob: Tensor，形状 `(B,)`，表示 $\log \pi(a\mid s)$（动作维度求和）。
            entropy: Tensor，形状 `(B,)`，表示策略熵（动作维度求和）。

        用法:
            `PPO.train()` 内部会调用它来计算 ratio/entropy/critic loss。
        """
        p = self.forward(state)
        # 均值策略(与 act 的 deterministic 输出保持一致)
        action_mean = torch.tanh(self.actor(p)) * self.max_action

        # 对角高斯策略(各维独立)，logprob/entropy 在动作维 sum
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)

        action_logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        # Critic输出
        values = self.critic(p).squeeze(-1)

        return values, action_logprob, entropy


class PPO(object):
    """
    简化版PPO
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim,
        discount=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        lr=3e-4,
        K_epochs=80,
        policy_noise=0.2,
    ):
        """初始化 PPO 算法对象。

        Args:
            state_dim: 状态维度。
            action_dim: 动作维度。
            max_action: 动作幅值上限。
            hidden_dim: 网络隐藏层宽度。
            discount: 折扣因子 $\gamma$。
            gae_lambda: GAE 的 $\lambda$。
            eps_clip: PPO clip 的 $\epsilon$。
            lr: 学习率。
            K_epochs: 每次更新时重复优化的轮数。
            policy_noise: 初始策略 std（初始化 log_std）。

        用法:
            通常配合 `OnPolicyRolloutBuffer` 使用：
            1) 交互时调用 `select_action_with_logprob()` 收集 (s,a,r,done,logp,next_s)
            2) 收集一段 rollout 后调用 `train(buffer)` 更新网络。
        """
        self.actorcritic = ActorCritic(
            state_dim,
            action_dim,
            hidden_dim,
            max_action,
            policy_noise=policy_noise,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.actorcritic.parameters(), lr=lr)

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 各loss系数
        self.actor_loss_coeff = 1.0
        self.critic_loss_coeff = 0.74
        self.entropy_loss_coeff = 0.01

    def select_action(self, state):
        """与环境交互时选择动作（默认探索采样）。

        Args:
            state: numpy 数组或可转为 tensor 的数组，形状 `(state_dim,)`。

        Returns:
            numpy.ndarray: 动作，形状 `(action_dim,)`。

        说明:
            这是便捷封装；训练时更推荐用 `select_action_with_logprob()` 以便写入 old_logprob。
        """
        return self.select_action_with_logprob(state, deterministic=False)[0]

    def select_action_with_logprob(self, state, deterministic=False):
        """选择动作并返回其在当前策略下的 logprob。

        Args:
            state: numpy 数组或可转为 tensor 的数组，形状 `(state_dim,)`。
            deterministic: 是否确定性输出。
                - True: 用均值动作（logprob 仍按该动作计算）。
                - False: 从高斯采样动作。

        Returns:
            action: numpy.ndarray，形状 `(1, action_dim)`（当前实现保留 batch 维）。
            logprob: numpy.ndarray，形状 `(1,)`。

        用法:
            在采样阶段将 `logprob` 写入 on-policy buffer，作为 PPO 更新时的 `old_logprob`。
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
        with torch.no_grad():
            p = self.actorcritic.forward(state_t)
            action_mean = torch.tanh(self.actorcritic.actor(p)) * self.actorcritic.max_action
            std = self.actorcritic.log_std.exp().expand_as(action_mean)
            dist = Normal(action_mean, std)
            if deterministic:
                action_t = action_mean
                logprob_t = dist.log_prob(action_t).sum(-1)
            else:
                action_t = dist.sample()
                logprob_t = dist.log_prob(action_t).sum(-1)

        return action_t.cpu().numpy(), logprob_t.cpu().numpy()

    def train(self, replay_buffer):
        """执行一次 PPO 更新（使用一段 on-policy rollout）。

        Args:
            replay_buffer: 需要实现 `on_policy_sample()`，返回：
                - 5 项：`(state, action, reward, not_done, next_state)`
                - 或 6 项：再加 `old_logprob`

        Returns:
            None

        训练逻辑:
            - 用 bootstrap 的 `V(next_state)` 计算 TD residual
            - 反向递推得到 GAE advantages，并计算 returns
            - PPO clip actor loss + critic MSE + entropy bonus
        """
        batch = replay_buffer.on_policy_sample()
        if len(batch) == 5:
            state, action, reward, not_done, next_state = batch
            old_logprob = None
        elif len(batch) == 6:
            state, action, reward, not_done, next_state, old_logprob = batch
        else:
            raise ValueError(
                "replay_buffer.on_policy_sample() 需要返回 5 或 6 个值: "
                "(state, action, reward, not_done, next_state[, old_logprob])"
            )

        # 统一 tensor / device
        state = state.to(device) if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.float32, device=device)
        next_state = (
            next_state.to(device)
            if torch.is_tensor(next_state)
            else torch.as_tensor(next_state, dtype=torch.float32, device=device)
        )
        action = action.to(device) if torch.is_tensor(action) else torch.as_tensor(action, dtype=torch.float32, device=device)
        reward = reward.to(device) if torch.is_tensor(reward) else torch.as_tensor(reward, dtype=torch.float32, device=device)
        not_done = not_done.to(device) if torch.is_tensor(not_done) else torch.as_tensor(not_done, dtype=torch.float32, device=device)

        # === GAE-\lambda: 计算 advantages 和 returns ===
        # 说明：这里假设 buffer 内是按时间顺序的一条(或多条拼接的)轨迹。
        # 当前实现的 train_test.py 是“每个 episode 一个 buffer”，因此天然满足顺序要求。
        with torch.no_grad():
            values = self.actorcritic.critic(self.actorcritic.forward(state)).squeeze(-1)  # V(s_t)
            next_values = self.actorcritic.critic(self.actorcritic.forward(next_state)).squeeze(-1)  # V(s_{t+1})

            deltas = reward + self.discount * not_done * next_values - values

            advantages = torch.zeros_like(values)
            gae = 0.0
            for t in reversed(range(values.shape[0])):
                gae = deltas[t] + self.discount * self.gae_lambda * not_done[t] * gae
                advantages[t] = gae

            returns = advantages + values

            # 标准化优势，提升训练稳定性
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 旧策略 logprob(固定，不参与后续 K_epochs 更新的梯度)
        if old_logprob is None:
            with torch.no_grad():
                _, old_logprob, _ = self.actorcritic.evaluate(state, action)
        else:
            old_logprob = old_logprob.to(device) if torch.is_tensor(old_logprob) else torch.as_tensor(old_logprob, dtype=torch.float32, device=device)

        old_logprob = old_logprob.detach()

        for _ in range(self.K_epochs):
            # 新策略
            values_new, logprob, dist_entropy = self.actorcritic.evaluate(state, action)

            # PPO clip目标
            ratio = torch.exp(logprob - old_logprob)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = F.mse_loss(values_new, returns)
            entropy_loss = dist_entropy.mean()

            loss = (self.actor_loss_coeff * actor_loss +
                    self.critic_loss_coeff * critic_loss -
                    self.entropy_loss_coeff * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, filename):
        """保存模型参数与优化器状态。

        Args:
            filename: 保存路径前缀（会写入 `filename` 与 `filename + "_optimizer"`）。

        Returns:
            None
        """
        torch.save(self.actorcritic.state_dict(), filename)
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        """加载模型参数与优化器状态。

        Args:
            filename: 保存路径前缀（读取 `filename` 与 `filename + "_optimizer"`）。

        Returns:
            None
        """
        self.actorcritic.load_state_dict(torch.load(filename))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))

    def eval_mode(self):
        """切换到评估模式（关闭 dropout/bn 等训练行为）。"""
        self.actorcritic.eval()

    def train_mode(self):
        """切换到训练模式。"""
        self.actorcritic.train()
