# getInitialDMP（如 initialDMP）
# IK（如 calculate_q）
# algos（DDPG/PPO/TD3）
# utils.memory（经验回放 ReplayBuffer）
# buffer（MemoryBuffer）

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from getInitialDMP import initialDMP
from movement_primitives.dmp._dmp import DMP
from IK import calculate_q
from gym.spaces import Box
import newton
from mpm_terrain import Example as TerrainSimulator

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class LegEnv(object):
    """单腿-混杂地形交互环境"""
    viewer = None
    dt = 
    action_bound = []
    goal = 
    state_dim = 4
    action_dim = 3
    _max_episode_steps = 100

    def __init__(self):
        # 单腿活动范围
        # 考虑是不是加入水平旋转关节
        self.action_space = Box(low=-self.action_bound[-1], high=self.action_bound[-1], shape=(self.action_dim,), dtype=np.float32)
        
        # 单腿信息：长度，角度
        self.leg_info = np.zeros(2, dtype=[('length', np.float32), ('theta', np.float32)])  
        self.leg_info['length'] = 20  # 初始长度
        self.leg_info['theta'] = np.pi / 4  # 初始角度
        self.center_coord = np.array([])

        self.finishFlag = np.array([])

        # 建立混杂地形
        self.terrain_simulator = TerrainSimulator()

        # 获得参考轨迹
        self.dmp = initialDMP()

    def step(self, action):
        done = False

        action = np.clip(action, *self.action_bound)

        # 施加动作至轨迹
        force_eta = action * 1000
        self.position, self.velocity, force_eta = self.dmp.step(
            self.position, self.velocity, force_eta = force_eta, finishFlag = self.finishFlag)
        
        # 如果超过轨迹的执行时间，则结束
        if self.dmp.t >= self.dmp.execution_time:
            done = True
        
        # 获得当前状态交互力
        F_N, F_T = self.terrain_simulator.get_contact_forces()

        r = 0

        F_N_max = 400
        F_T_max = 400
        if (F_N > F_N_max):
            self.finishFlag[0] = 1
        else:
            self.finishFlag[0] = 0
        if (F_T > F_T_max):
            self.finishFlag[1] = 1
        else:
            self.finishFlag[1] = 0
        
        if(F_N > F_N_max and F_T > F_T_max):
            done = True
            r = F_N + F_T

        disToCenter = np.linalg.norm(self.position - self.center_coord)
        if disToCenter > 180:
            done = True
            r = -50
        
        # 构造state向量，包含末端位置和接触力信息
        # 归一化，神经网络对输入数据的初度比较敏感，大数将主导学习过程
        s = np.concatenate((self.position / 200.0, np.array([F_N / 10]), np.array([F_T / 10])))

        return s, r, done
    
    def reset(self):
        # 随机化目标点

        # 重新建立混杂地形
        self.terrain_simulator = TerrainSimulator()

        # 随机化单腿初始姿态
        self.leg_info['theta'] = 2 * np.pi * np.random.rand(2)

        # 重置动态运动基元（DMP）
        self.dmp.reset()

        # 更新可视化窗口（如果存在）
        # 根据实际接口修改
        if self.viewer is not None:
            self.viewer.reset()

        # 构建初始状态
        self.position = 
        s = np.concatenate(())

        return s
    
    def render(self):
        pass

    def sample_action(self):
        pass

class viewer():

    def __init__(self, env):
        pass

    def resetViewer(self):
        pass

    def render(self):
        pass

    def _update_leg(self, leg_info):
        pass

if __name__ == "__main__":
    env = LegEnv()
    while True:
        env.render()
        env.step(env.sample_action())
