# import pyglet
# import numpy as np
# class LegEnv(object):
#     viewer = None
#     dt = .1    # refresh rate
#     action_bound = [-1, 1]
#     goal = {'x': 100., 'y': 100., 'l': 40}
#     state_dim = 9
#     action_dim = 2

#     def __init__(self):
#         self.leg_info = np.zeros(
#             2, dtype=[('l', np.float32), ('r', np.float32)])
#         self.leg_info['l'] = 100        # 2 arms length
#         self.leg_info['r'] = np.pi/6    # 2 angles information
#         self.on_goal = 0

#     def step(self, action):
#         done = False
#         action = np.clip(action, *self.action_bound)
#         self.leg_info['r'] += action * self.dt
#         self.leg_info['r'] %= np.pi * 2    # normalize

#         (a1l, a2l) = self.leg_info['l']  # radius, arm length
#         (a1r, a2r) = self.leg_info['r']  # radian, angle
#         a1xy = np.array([200., 200.])    # a1 start (x0, y0)
#         a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
#         finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
#         # normalize features
#         dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
#         dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
#         r = -np.sqrt(dist2[0]**2+dist2[1]**2)

#         # done and reward
#         if (self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2
#         ) and (self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2):
#             r += 1.
#             self.on_goal += 1
#             if self.on_goal > 50:
#                 done = True
#         else:
#             self.on_goal = 0

#         # state
#         s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
#         return s, r, done

#     def reset(self):
#         self.leg_info['r'] = 2 * np.pi * np.random.rand(2)
#         self.on_goal = 0
#         (a1l, a2l) = self.leg_info['l']  # radius, arm length
#         (a1r, a2r) = self.leg_info['r']  # radian, angle
#         a1xy = np.array([200., 200.])  # a1 start (x0, y0)
#         a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
#         finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
#         # normalize features
#         dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
#         dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
#         # state
#         s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
#         return s

#     def render(self):
#         if self.viewer is None:
#             self.viewer = Viewer(self.leg_info, self.goal)
#         self.viewer.render()

#     def sample_action(self):
#         return np.random.rand(2)-0.5    # two radians

# class Viewer(pyglet.window.Window):
#     bar_thc = 5
#     def __init__(self, leg_info, goal):
#         super(Viewer, self).__init__(width=400, height=400, resizable=False, caption="Leg",vsync=False)
#         pyglet.gl.gl.glClearColor(1, 1, 1, 1)

#         self.leg_info = leg_info
#         self.center = np.array([200, 200])

#         self.batch = pyglet.graphics.Batch()
        
#         # 使用 Rectangle 绘制目标
#         self.goal = pyglet.shapes.Rectangle(
#             goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,
#             goal['l'], goal['l'],
#             color=(86, 109, 249),
#             batch=self.batch
#         )
        
#         # 为机械臂创建顶点列表
#         self.leg1_vertices = [250.0, 250.0, 250.0, 300.0, 260.0, 300.0, 260.0, 250.0]
#         self.leg2_vertices = [100.0, 150.0, 100.0, 160.0, 200.0, 160.0, 200.0, 150.0]

#     def render(self):
#         self._update_view_leg()
#         self.switch_to()
#         self.dispatch_events()
#         self.dispatch_event('on_draw')
#         self.flip()

#     def on_draw(self):
#         self.clear()
#         self.batch.draw()
        
#         # 手动绘制机械臂 - 使用 pyglet.shapes
#         leg1_shape = pyglet.shapes.Polygon(
#             *[(self.leg1_vertices[i], self.leg1_vertices[i+1]) for i in range(0, len(self.leg1_vertices), 2)],
#             color=(249, 86, 86)
#         )
#         leg1_shape.draw()
        
#         leg2_shape = pyglet.shapes.Polygon(
#             *[(self.leg2_vertices[i], self.leg2_vertices[i+1]) for i in range(0, len(self.leg2_vertices), 2)],
#             color=(249, 86, 86)
#         )
#         leg2_shape.draw()

#     def _update_view_leg(self):
#         (a1l, a2l) = self.leg_info['l']     # radius, arm length
#         (a1r, a2r) = self.leg_info['r']     # radian, angle
#         a1xy = self.center            # a1 start (x0, y0)
#         a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
#         a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

#         a1tr, a2tr = np.pi / 2 - self.leg_info['r'][0], np.pi / 2 - self.leg_info['r'].sum()
#         xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
#         xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
#         xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
#         xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

#         xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
#         xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
#         xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
#         xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

#         self.leg1_vertices = np.concatenate((xy01, xy02, xy11, xy12)).tolist()
#         self.leg2_vertices = np.concatenate((xy11_, xy12_, xy21, xy22)).tolist()


# if __name__ == "__main__":
#     env = LegEnv()
#     while True:
#         # s = env.reset()
#         # for i in range(400):
#             env.render()
#             env.step(env.sample_action())

import numpy as np
import pyglet
from forceFeild import gaussianField
import matplotlib.pyplot as plt
from getInitialDMP import initialDMP
from movement_primitives.dmp._dmp import DMP 
from IK import calculate_q
import pickle
import time
from gym.spaces import Box
from contactModelField import Arrow, ContactForceField, ArrowDraw
import math

class ArmEnv(object):
    """
    定义了机械臂与地形交互的强化学习环境。
    该环境模拟了机械臂末端（足端）在未知地形上移动的过程，
    智能体通过调整DMP（动态运动基元）轨迹来学习如何控制接触力。
    """
    viewer = None # 可视化查看器实例
    dt = .1    # 模拟的时间步长 (refresh rate)
    action_bound = [-1, 1] # 归一化的动作值范围
    goal = {'x': 100., 'y': 100., 'l': 40} # 目标点信息 (当前未使用)
    state_dim = 4 # 状态空间的维度: [x, y, 法向力, 切向力]
    action_dim = 2 # 动作空间的维度，用于调整DMP轨迹的力项
    _max_episode_steps = 100 # 每个回合的最大步数
    
    # You would also


    def __init__(self):
        """
        环境初始化函数。
        """
        # 定义一个连续动作空间，动作是 2 维浮点向量，且每个分量范围均在 [-1, 1] 之间。
        self.action_space = Box(low=-self.action_bound[-1], high=self.action_bound[-1], shape=(self.action_dim,), dtype=np.float32)

        # 初始化机械臂物理信息
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 2个臂节的长度
        self.arm_info['r'] = np.pi/6    # 2个关节的初始角度信息
        self.center_coord = np.array([200, 300]) # 机械臂基座在世界坐标系中的坐标

        # 完成标志，用于指示力是否超出阈值，[法向力标志, 切向力标志]
        self.finishFlag = np.array([0,0])
        # 建立接触力势场
        self.size = 400 # 势场区域的大小
        self.contactForceField = ContactForceField(self.size)

        # 获取参考轨迹 (通过DMP)
        self.dmp = initialDMP() # 初始化DMP，生成一条基准轨迹
        self.position = self.dmp.start_y # 末端执行器的初始位置，从DMP的起点获取
        self.velocity = np.zeros_like(self.position) # 末端执行器的初始速度

      
    def step(self, action):
        """
        在环境中执行一个时间步。
        :param action: 智能体输出的动作。
        :return: (state, reward, done, info) 元组。
        """
        done = False # 回合结束标志
        # 将动作裁剪到定义的边界内，防止异常值。
        action = np.clip(action, *self.action_bound)   # normalize action to [-1, 1] 
        
        # 将智能体的动作（通常是-1到1之间）转换为施加到DMP上的外部力。
        # 这个力会修改原始的DMP轨迹。系数500是一个需要调整的超参数。
        force_eta =  action * 500 # 系数需要调参
        # print(force_eta)
        # 执行DMP一步，更新末端执行器的位置和速度。
        # force_eta是学习到的力，finishFlag用于告知DMP力是否超限。
        self.position, self.velocity = self.dmp.step(
            self.position, self.velocity, force_eta = force_eta, finishFlag = self.finishFlag)
        
        # 如果DMP轨迹的执行时间结束，则认为这个回合自然完成。
        if self.dmp.t >= self.dmp.execution_time_:
            done = True  # 这里不行，需要更改

        # 根据当前末端位置，从接触力场中获取法向力(F_N)和切向力(F_T)。
        F_N, F_T = self.contactForceField.getForceByPosition(self.position)

        # 计算总接触力的幅值 (当前未使用)
        aa = math.sqrt(F_N ** 2 + F_T ** 2)
        
        finger = self.position # 末端位置的别名
        F_finger = np.array([aa]) # 总接触力幅值 (当前未使用)

        r = 0 # 初始化奖励
        

        # 定义力的阈值，用于判断接触是否稳定。
        F_N_max = 400
        F_T_max = 400
        # print(finger)
        # print(F_finger)
        # 如果法向力或切向力超出阈值，设置相应的标志位。
        # 这个标志位会传递给DMP的下一步，可能用于调整轨迹生成。
        if(F_N > F_N_max):
            self.finishFlag[0] = 1
        else:
            self.finishFlag[0] = 0
        if(F_T > F_T_max):
            self.finishFlag[1] = 1
        else:
            self.finishFlag[1] = 0


        # 如果法向力和切向力都超出阈值，则认为任务失败，回合结束，并给予惩罚。
        if(F_N > F_N_max and F_T > F_T_max):
            done = True
            r =  F_N + F_T # 惩罚值等于两种力的和

        # 判断是否超过运动学范围。如果末端超出最大伸展范围，任务失败，给予较大惩罚。
        disToCenter = np.linalg.norm(finger - self.center_coord)
        if disToCenter > 180: # 2*arm_length -20 (留有一定余量)
            done = True
            r = -50

        # 构造状态向量，包含末端位置和接触力信息。
        # 对状态进行归一化，以利于神经网络学习。
        s = np.concatenate((finger/200,  np.array([F_N])/10, np.array([F_T])/10))  # state(末端位置, 法向力, 切向力)

        return s, r, done, None

    def reset(self):
        """
        重置环境到初始状态。
        :return: 初始状态。
        """
        # 随机化目标点 (当前未使用)
        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400. # random goal
        
        ## 重新初始化接触力势场，使得每次回合的地形都不同。
        self.contactForceField = ContactForceField(self.size)

        
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)   # 随机化机械臂的初始角度
        self.dmp.reset() # 重置DMP到其初始状态

        # 如果可视化窗口已创建，则用新的力场重置它。
        if self.viewer is not None:
            self.viewer.resetViewer(self.contactForceField)

        finger = self.dmp.start_y # 获取DMP的初始位置作为末端位置
        
        # 构造初始状态向量，此时接触力为0。
        s = np.concatenate((finger/200,  [0], [0]))
        return s

    def render(self):
        """
        渲染当前环境状态。
        """
        # 如果可视化窗口不存在，则创建一个。
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal, self.contactForceField, self.center_coord)
        # 通过逆运动学计算关节角度以匹配末端位置。
        (a1l, a2l) = self.arm_info['l'] # 臂长
        a1xy = self.viewer.center_coord    # 基座坐标
        finger = self.position # 当前末端位置

        finger_RbtFrame = finger - a1xy # 将末端位置转换到以基座为原点的坐标系
        # 调用逆运动学函数计算两个关节的角度q1, q2。
        q1, q2 = calculate_q(a1l, a2l, finger_RbtFrame[0], finger_RbtFrame[1])
        if(q1 == None or q2 == None): # 如果逆解不存在（位置不可达），则不更新渲染。
            return
        # 更新机械臂的关节角度信息。
        self.arm_info['r'][0] = q1
        self.arm_info['r'][1] = q2
        
        # 调用viewer的render方法来绘制机械臂和力场。
        self.viewer.render(self.position)

    def sample_action(self): #随机动作
        """
        从动作空间中随机采样一个动作，用于测试。
        """
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):  # pyglet 可视化模块
    """
    使用 pyglet 的可视化查看器类。
    负责绘制机械臂、接触力场、力矢量和可达范围。
    """
    bar_thc = 5 # 机械臂的厚度

    def __init__(self, arm_info, goal, contactForceField, center_coord):
        """
        查看器初始化。
        :param arm_info: 机械臂信息 (臂长、角度)。
        :param goal: 目标信息 (当前未使用)。
        :param contactForceField: 接触力场对象。
        :param center_coord: 机械臂基座坐标。
        """
        # vsync=False: 不垂直同步，可以加速训练时的渲染。
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)  # vsync=False 是否与屏幕刷新率同步
        pyglet.gl.glClearColor(1, 1, 1, 1)   # 设置背景颜色为白色
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = center_coord

        self.mouseX = 0
        self.mouseY = 0

        self.resetViewer(contactForceField)
        

        
        # self.forceField = forceField

    def resetViewer(self, contactForceField):
        """
        用新的力场数据重置查看器。
        :param contactForceField: 新的接触力场对象。
        """
        self.contactForceField = contactForceField
        # 从力场对象获取网格坐标和力分量，用于可视化。
        self.Field_X, self.Field_Y, self.F_N, self.F_T = contactForceField.getPotentialField()


        self.batch = pyglet.graphics.Batch()    # 创建一个批处理对象，用于高效绘制大量图形元素。

        # 遍历力场的每个网格点，创建一个箭头来表示该点的力。
        [x_length,y_length] = self.Field_X.shape
        for i in range(x_length):
            for j in range(y_length):
                # 计算力的方向和大小
                angle = math.atan2(self.F_N[i, j], self.F_T[i, j]) * 180 / math.pi
                length = math.sqrt(self.F_N[i, j] ** 2 + self.F_T[i, j] ** 2)*0.05 # 缩放以适应窗口
                # 创建箭头对象并添加到批处理中
                arrow = Arrow(self.batch ,self.Field_X[i, j], self.Field_Y[i, j], angle, length)
                arrow.draw()


        # 创建两个四边形代表机械臂的两个臂节，并添加到批处理中。
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # 初始顶点位置 (将被_update_arm更新)
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # 颜色
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # 初始顶点位置
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        # self._update_field(forceField)




        


    def render(self, footPosition):
        """
        主渲染函数，在每个时间步被调用。
        :param footPosition: 机械臂末端（足端）的当前位置。
        """
        
        self._update_arm() # 根据最新的关节角度更新机械臂的顶点位置。
        self.switch_to() # 切换OpenGL上下文到当前窗口。
        self.dispatch_events() # 处理所有待处理的窗口事件（如关闭窗口）。
        self.dispatch_event('on_draw') # 手动触发 on_draw 事件，调用 on_draw 方法。
        
        # print(footPosition)

        # 绘制末端执行器当前位置所受的力。
        F_N, F_T = self.contactForceField.getForceByPosition(footPosition)
        # 绘制合力箭头
        angle = math.atan2(F_N, F_T) * 180 / math.pi
        length = math.sqrt(F_N ** 2 + F_T ** 2)*0.05
        arrowD = ArrowDraw(footPosition[0], footPosition[1], angle=angle, length=length)
        arrowD.draw()
        # 绘制切向力分量 (x方向)
        arrowD_x = ArrowDraw(footPosition[0], footPosition[1], angle=0, length=F_T*0.05)
        arrowD_x.draw()
        # 绘制法向力分量 (y方向)
        arrowD_normal = ArrowDraw(footPosition[0], footPosition[1], angle=90, length=F_N*0.05)
        arrowD_normal.draw()

        self.flip() # 交换前后缓冲区，将绘制的内容显示在屏幕上。

        # print(F_N, F_T)
    
    def draw_reach_boundary(self, center, radius, segments=100):
        """
        绘制一个圆圈，表示机械臂的最大可达范围。
        :param center: 圆心坐标。
        :param radius: 圆的半径。
        :param segments: 用于绘制圆的线段数量。
        """
        verts = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            verts += [x, y]

        # 使用pyglet的底层绘图API绘制线圈。
        pyglet.graphics.draw(len(verts)//2, pyglet.gl.GL_LINE_LOOP, ('v2f', verts), ('c3B', (150,150,150) * (len(verts)//2)))

    def on_draw(self):
        """
        pyglet的绘图事件回调函数。
        """
        self.clear() # 清除窗口内容。
        self.batch.draw() # 绘制批处理中的所有图形元素（力场箭头、机械臂）。
        self.draw_reach_boundary(center=self.center_coord, radius=180) #调用新增函数 

    

    def _update_arm(self):
        """
        根据机械臂的当前关节角度，计算并更新两个臂节的顶点坐标。
        """

        # update arm
        (a1l, a2l) = self.arm_info['l']     # 臂长
        (a1r, a2r) = self.arm_info['r']     # 关节角度
        a1xy = self.center_coord            # 臂1的起始点 (基座)
        # 计算臂1的终点，也是臂2的起点
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy
        # 计算臂2的终点 (末端执行器)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_

        # --- 以下为计算臂节四边形顶点的复杂几何变换 ---
        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        # 更新臂节的顶点数据
        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

        



    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        """
        鼠标移动事件回调，记录鼠标位置 (当前未使用)。
        """
        self.mouseX = x
        self.mouseY = y


if __name__ == '__main__':
    # 如果此文件作为主程序运行，则创建一个环境实例并无限循环执行。
    # 在每个循环中，渲染环境并执行一个随机动作。
    # 这对于测试环境是否正常工作很有用。
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())