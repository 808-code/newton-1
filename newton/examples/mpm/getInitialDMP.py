from movement_primitives.dmp._dmp import DMP 
import numpy as np
import matplotlib.pyplot as plt

def initialDMP():

    # 定义指数函数的参数
    a = 2.0  # 底数
    b = 4.3  # 指数

    # 生成x轴上的数据点
    T = np.linspace(0, 1, 100)  # 在区间[0, 10]内生成100个数据点
    x = np.linspace(1, 0, 100)
    # 计算指数函数的值
    y = a * np.exp(b * x)


    Y = np.column_stack((x, y))

    start = Y[0]
    goal = Y[-1]


    dmp = DMP(n_dims=len(start), execution_time=1.0, dt=0.01, n_weights_per_dim=10,
            smooth_scaling=True)

    # 模拟指数函数的轨迹生成DMP
    dmp.imitate(T, Y)
    dmp.configure(start_y=np.array([263., 200.]), goal_y=np.array([173., 166.]))
    return dmp
