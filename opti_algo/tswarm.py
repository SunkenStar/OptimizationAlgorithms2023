import random
import numpy as np
from opti_algo.skeleton import Optimizer


class _Particle:
    def __init__(self, dimension, zone, c1, c2, c_tabu):
        """
        粒子类。\n
        参数说明：\n
        dimension：问题维度，或者说自变量个数\n
        zone：长度等于dimension的列表，列表中每个元素为对应位置自变量的取值范围\n
        c1、c2：粒子群优化算法的参数
        c_tabu：禁忌产生的斥力系数
        """
        self.x = []
        self.v = []
        self.dimension = dimension
        for i in range(dimension):
            self.x.append(random.uniform(zone[i][0], zone[i][1]))
            self.v.append(self.x[i] * random.random())
        self.c1 = c1
        self.c2 = c2
        self.c_tabu = c_tabu
        self.err = np.inf
        self.err_best = np.inf
        self.x_best = self.x.copy()
        self.zone = zone

    def get_fitness(self, target_func):
        """
        计算粒子的适应度。这里用到的是“误差”err，或者说优化目标是取target_func的最小值
        """
        self.err = target_func(self.x)
        if self.err < self.err_best:
            self.err_best = self.err
            self.x_best = self.x.copy()

    def update(self, g, tabu_list):
        """
        粒子按照\n
        x = x + v\n
        v = 0.5v + c1*r1*(p-x) + c2*r2*(g-x) - c_tabu*r3*(t-x)\n
        移动\n
        也即考虑自身惯性0.5v、历史最优的影响、群体中同伴的影响以及禁忌表中位置的影响。\n
        禁忌表中每一项禁忌都会产生一个和距离成反比的“斥力”影响粒子，这样可以半强迫式地要求算法搜索其他区域。\n
        当粒子碰撞边界时速度变为原来的反向，位置取在边界上
        """
        for i in range(self.dimension):
            self.x[i] = self.x[i] + self.v[i]
            if self.x[i] > self.zone[i][1]:
                self.x[i] = self.zone[i][1]
                self.v[i] = -self.v[i]
            if self.x[i] < self.zone[i][0]:
                self.x[i] = self.zone[i][0]
                self.v[i] = -self.v[i]
        for i in range(self.dimension):
            history_influence = self.c1 * random.random() * (self.x_best[i] - self.x[i])
            peer_influence = self.c2 * random.random() * (g[i] - self.x[i])
            tabu_influence = 0
            for tabu in tabu_list:
                if tabu:
                    tabu_influence += (
                        self.c_tabu * random.random() * (self.x[i] - tabu[i])
                    )
            self.v[i] = (
                0.5 * self.v[i] + history_influence + peer_influence + tabu_influence
            )


class TSwarm(Optimizer):
    """
    禁忌粒子群算法，求取目标函数最小值\n
    在传统粒子群搜索算法中结合了禁忌搜索的思想。\n
    定义五位禁忌表，采用队列的数据结构，每次迭代后入队当前最优位置，相应使得一个位置出队\n
    禁忌表中每个位置对粒子产生一个和距离成比例的斥力，强迫搜索未探索部分。\n
    args可选传入以下参数按顺序排成的元组：\n
        c1、c2：粒子群优化算法的参数
        c_tabu：禁忌产生的斥力系数
        particle_num：粒子个数
        max_iter：迭代次数上限
    """

    def __str__(self):
        return "禁忌粒子群算法"

    def optimize(self):
        if self.args:
            arguments = self.args
        else:
            arguments = (1, 3, 0.4, 40, 200)
        c1, c2, c_tabu, particle_num, maxiter = arguments
        swarm = []
        for _ in range(particle_num):
            swarm.append(_Particle(self.dimension, self.zone, c1, c2, c_tabu))
        global_best = np.inf
        g = []
        tabu_list = [[]] * 5
        for i in range(maxiter):
            for p in swarm:
                p.get_fitness(self.target_func)
                if p.err_best < global_best:
                    global_best = p.err_best
                    g = p.x_best.copy()
            for p in swarm:
                p.update(g, tabu_list)
            tabu_list.pop(0)
            tabu_list.append(g.copy())
            self.searching_history.append((g.copy(), global_best))
        return self.searching_history[-1]


if __name__ == "__main__":

    def sphere(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    tswarm_algo = TSwarm(sphere, 2, [(-10, 10), (-10, 10)])
    print(tswarm_algo, tswarm_algo.optimize())
    tswarm_algo.visualize()
