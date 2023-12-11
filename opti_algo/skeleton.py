from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Optimizer(ABC):
    """
    智能优化算法通用抽象基类，为2023秋计算智能算法课程期末作业设计。\n
    """

    def __init__(self, target_func, dimension, zone, args=None):
        """
        参数说明：\n
        target_func：目标函数\n
        dimension：问题维度，或者说自变量个数\n
        zone：[(a,b),(c,d)]格式的各维度取值范围列表
        args：算法特定参数
        """
        self.target_func = target_func
        self.dimension = dimension
        self.zone = zone
        self.args = args
        self.searching_history = [([], np.inf)]

    def visualize(self):
        """
        简单的可视化工具，绘制目标函数函数值随迭代次数的变化
        """
        best_err_history = [x[1] for x in self.searching_history]
        sns.set_style(style='white', rc={'font.sans-serif': ['FangSong', 'simhei']})
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'font.size': 10})
        plt.xlim(0, len(self.searching_history))
        plt.xlabel('迭代次数')
        plt.ylabel('当前迭代最优函数值')
        plt.plot(np.arange(1, len(self.searching_history)), np.array(best_err_history[1:]), color='k')
        plt.show()

    @abstractmethod
    def __str__(self):
        """
        算法名
        """

    @abstractmethod
    def optimize(self):
        """
        算法本体，返回[(最优坐标),最优值]
        """
        pass
     