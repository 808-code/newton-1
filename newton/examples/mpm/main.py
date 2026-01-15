import argparse # 解析命令行参数
import gym # OpenAI Gym，用于创建强化学习环境
import numpy as np
import os # 操作系统接口
# import pybullet_envs  # noqa F401 # 预加载 PyBullet 的 Gym 环境
# import pybulletgym  # noqa F401 register PyBullet enviroments with open ai gym
import torch # PyTorch，深度学习框架
import time # 时间控制模块
from algos import DDPG, PPO, TD3 # 导入自定义算法模块（DDPG、PPO、TD3）
from utils import memory # 导入经验回放缓冲区

from env import ArmEnv # 自定义机械臂环境
from buffer import MemoryBuffer  # 导入自定义的简化版回放缓冲区类

addSpeedForEvaluation = False # 是否加速评估过程，False表示不加速，会通过time.sleep减慢渲染

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# 评估策略性能函数
def eval_policy(policy, env, seed, eval_episodes=10, test=False):
    """
    评估一个给定的策略在环境中的表现。
    :param policy: 需要被评估的策略。
    :param env: 评估所用的环境。
    :param seed: 随机种子，确保评估的可复现性。
    :param eval_episodes: 评估所运行的回合（episode）数量。
    :param test: 是否为测试模式。
    :return: 在多个回合中的平均奖励。
    """
    policy.eval_mode() # 将策略切换到评估模式（不进行探索和学习）
    avg_reward = 0.  # 初始化累加奖励，用于计算多个 episode 的平均奖励

    # eval_episodes=100
    for _ in range(eval_episodes): # 循环指定次数的评估回合
        #if test:
        #    env.render(mode='human', close=False)
        state, done = env.reset(), False # 重置环境到初始状态，获得初始观测state
        hidden = None # 储存 RNN 的隐藏状态，对于非递归策略则为None
        while not done: # 在一个回合（episode）中循环，直到回合结束
            env.render() # 渲染当前环境状态，便于人眼观察 agent 的行为
            action, hidden = policy.select_action(np.array(state), hidden) # 根据当前策略选择确定性动作（无探索噪声）
            # env.render(mode='human', close=False)
            state, reward, done, _ = env.step(action) # 在环境中执行动作，获取反馈
            avg_reward += reward # 累加奖励
            if not addSpeedForEvaluation: # 如果不加速评估
                time.sleep(.06) # 暂停以减慢渲染速度，便于观察
        time.sleep(.5) # 每个回合结束后暂停0.5秒

    avg_reward /= eval_episodes # 计算所有评估回合的平均奖励

    policy.train_mode() # 将策略切换回训练模式
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

# 解析命令行参数、准备训练/评估环境和配置参数
def main():
    """
    主函数，负责解析参数、初始化环境/策略、执行训练循环和评估。
    """
    parser = argparse.ArgumentParser() # 创建一个命令行参数解析器，用于从命令行读取参数。
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="PPO", help="使用的策略名称 (例如: TD3, DDPG, PPO)")
    # OpenAI gym environment name
    parser.add_argument("--env", default="leg", help="环境名称")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int, help="设置 Gym, PyTorch 和 Numpy 的随机种子")
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=1e4, type=int, help="在训练开始时，执行随机动作探索的步数")
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int, help="评估策略的频率（以时间步为单位）")
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="训练的总最大时间步数")
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.45, help="添加到动作上的高斯探索噪声的标准差")
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=100, type=int, help="Actor和Critic网络训练时使用的批量大小")
    # Memory size
    parser.add_argument("--memory_size", default=1e6, type=int, help="经验回放缓冲区的最大容量")
    # Learning rate
    parser.add_argument("--lr", default=3e-4, type=float, help="学习率，太大会导致训练不稳定，太小收敛慢") #学习率，太大会导致训练不稳定，太小收敛慢
    # Discount factor
    parser.add_argument("--discount", default=0.99, help="折扣因子 γ，控制未来奖励的权重。越接近 1，表示越重视长期奖励") #折扣因子 γ，控制未来奖励的权重。越接近 1，表示越重视长期奖励
    # Target network update rate
    parser.add_argument("--tau", default=0.005, help="目标网络软更新的速率")
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.25, help="TD3中目标策略平滑所用噪声的标准差")
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5, help="TD3中目标策略平滑噪声的裁剪范围")
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int, help="TD3中策略（Actor）和目标网络延迟更新的频率")
    # Model width
    parser.add_argument("--hidden_size", default=256, type=int, help="神经网络隐藏层的神经元数量")
    # Use recurrent policies or not
    parser.add_argument("--recurrent", action="store_true", help="是否启用循环策略网络 (RNN/LSTM)") #是否启用循环策略网络
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true", help="是否在评估奖励提高时保存模型") #是否保存训练好的模型和优化器参数
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="", help="要加载的预训练模型文件名") #指定要加载的模型名
    # Don't train and just run the model
    parser.add_argument("--test", action="store_true", help="是否只进行测试而不训练") #测试模式
    args = parser.parse_args() # 解析所有定义的命令行参数，args.xxx 用于访问每个参数

    file_name = f"{args.policy}_{args.env}_{args.seed}" # 根据策略、环境和种子生成一个唯一的文件名，用于保存模型和结果
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"): # 如果 "results" 目录不存在，则创建它
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"): # 如果需要保存模型且 "models" 目录不存在，则创建它
        os.makedirs("./models")


    
    # set env
    env = ArmEnv()    # 创建自定义的机械臂环境实例
    # Set seeds
    #env.seed(args.seed)
    torch.manual_seed(args.seed) # 设置 PyTorch 的随机种子，以保证实验结果的可复现性
    np.random.seed(args.seed) # 设置 NumPy 的随机种子

    state_dim = env.state_dim # 从环境中获取状态空间的维度
    action_dim = env.action_dim # 从环境中获取动作空间的维度
    max_action = env.action_bound[-1] # 获取动作的最大值
    print(env.action_bound)

    args.recurrent = True #  强制使用递归神经网络结构（RNN），以处理时序依赖的状态信息

    # TODO: Add this to parameters
    recurrent_actor = args.recurrent # actor 是否使用递归结构
    recurrent_critic = args.recurrent # critic 是否使用递归结构

    print('recurrent', recurrent_critic)
    # 整理所有策略共享的初始化参数到一个字典中
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "hidden_dim": args.hidden_size,
        "discount": args.discount,
        "tau": args.tau,
        "recurrent_actor": recurrent_actor,
        "recurrent_critic": recurrent_critic,
    }

    # Initialize policy  根据命令行参数选择并初始化相应的RL策略
    if args.policy == "TD3":
        # 为TD3算法添加特有的参数
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    elif args.policy == "DDPG":
        # 初始化DDPG策略
        policy = DDPG.DDPG(**kwargs)

    elif args.policy == "PPO":
        # 为PPO算法添加特有的参数
        # TODO: Add kwargs for PPO
        kwargs["K_epochs"] = 10 # PPO中每次更新策略的迭代次数
        kwargs["eps_clip"] = 0.1 # PPO中用于裁剪优势函数的epsilon值
        policy = PPO.PPO(**kwargs)
        args.start_timesteps = 0 # PPO是on-policy算法，不需要初始的随机探索阶段
        n_update = 2048 # PPO中，每收集n_update步数据后进行一次策略更新


    if args.load_model != "": # 如果指定了要加载的模型
        policy_file = file_name \
            if args.load_model == "default" else args.load_model # 确定模型文件名
        policy.load(f"{policy_file}") # 加载模型参数

    if args.test: # 如果是测试模式
        eval_policy(policy, env, args.seed, eval_episodes=100, test=True) # 只执行评估并退出
        return
    # 创建经验回放缓冲区，用于存储和采样训练数据
    replay_buffer = memory.ReplayBuffer(
        state_dim, action_dim, args.hidden_size,
        args.memory_size, recurrent=recurrent_actor)

    # Evaluate untrained policy 评估未经训练的初始策略，作为基线
    evaluations = [eval_policy(policy, env, args.seed)]

    best_reward = evaluations[-1] # 将初始评估奖励设为当前最佳奖励
    # 初始化训练过程中的变量
    state, done = env.reset(), False # 重置环境，获取初始状态
    episode_reward = 0 # 当前回合的累计奖励
    episode_timesteps = 0 # 当前回合已进行的时间步数
    episode_num = 0 # 已完成的回合数
    hidden = policy.get_initial_states() # 获取RNN的初始隐藏状态

    # 创建一个大小为 5 的内存缓冲区 (当前未使用)
    memory_buffer = MemoryBuffer(max_size=5)


    # 主训练循环，从1循环到最大时间步数
    for t in range(1, int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps: # 在初始探索阶段
            action = env.action_space.sample() # 完全随机地选择动作
            _, next_hidden = policy.select_action(np.array(state), hidden) # 仍然需要通过策略网络来更新隐藏状态
        else: # 探索阶段结束后
            a, next_hidden = policy.select_action(np.array(state), hidden) # 使用策略网络选择动作
            action = (
                a + np.random.normal( # 为动作添加高斯噪声以进行探索
                    0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action) # 将添加噪声后的动作裁剪到有效范围内
            print("a", a)
            print("action", action)

        # Perform action
        env.render() # 渲染环境


        next_state, reward, done, _ = env.step(action) # 在环境中执行动作，并获取下一个状态、奖励、结束标志等信息
        # memory_buffer.add(force)
        # reward = memory_buffer.calculate_slope() + 0.5*force  # !!!!!!!!!!!!!!!!!!!!!!!!!!!

        # 判断回合是否真正结束。如果是因为达到最大步数而结束，done_bool为0，表示这不是一个终止状态。
        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        # 将(状态, 动作, 下一状态, 奖励, 结束标志, 隐藏状态, 下一隐藏状态)的转换元组存入经验回放区
        replay_buffer.add(
            state, action, next_state, reward, done_bool, hidden, next_hidden)
        # 更新当前状态和隐藏状态
        state = next_state
        hidden = next_hidden
        episode_reward += reward # 累加当前回合的奖励

        # Train agent after collecting sufficient data 根据策略类型（on-policy/off-policy）进行训练
        if (not policy.on_policy) and t >= args.start_timesteps: # 对于off-policy算法（如TD3, DDPG），在探索期结束后每一步都训练
            policy.train(replay_buffer, args.batch_size)
        elif policy.on_policy and t % n_update == 0: # 对于on-policy算法（如PPO），每收集n_update步数据后进行一次训练
            policy.train(replay_buffer)
            replay_buffer.clear_memory() # on-policy算法训练后需要清空缓冲区

        if done: # 如果一个回合结束
            # +1 to account for 0 indexing. +0 on ep_timesteps since it
            #  will increment +1 even if done=True
            # 打印当前回合的统计信息
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False # 重置环境，开始新回合
            episode_reward = 0 # 重置回合奖励
            episode_timesteps = 0 # 重置回合步数
            episode_num += 1 # 回合数加一
            hidden = policy.get_initial_states() # 重置 RNN 隐状态
            memory_buffer.clear() # 清空辅助缓冲区

        # Evaluate episode 定期评估策略并保存模型
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, env, args.seed)) # 执行评估
            if evaluations[-1] > best_reward and args.save_model: # 如果当前评估奖励超过历史最佳奖励，并且设置了保存模型
                policy.save(f"./models/{file_name}") # 保存当前策略模型

            np.save(f"./results/{file_name}", evaluations) # 保存所有评估奖励历史，用于后续分析


if __name__ == "__main__":
    main() # 如果该脚本作为主程序运行，则执行main函数