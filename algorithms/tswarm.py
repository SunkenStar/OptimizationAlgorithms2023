import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Particle:
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
        self.err = target_func(self.x)
        if self.err < self.err_best:
            self.err_best = self.err
            self.x_best = self.x.copy()

    def update(self, g, tabu_list):
        """
        粒子按照\n
        x = x + v\n
        v = 0.5v + c1*r1*(p-x) + c2*r2*(g-x)\n
        移动\n
        当粒子碰撞边界时速度变为原来的反向，位置取在边界上
        """
        for i in range(len(self.x)):
            self.x[i] = self.x[i] + self.v[i]
            if self.x[i] > self.zone[i][1]:
                self.x[i] = self.zone[i][1]
                self.v[i] = - self.v[i]
            if self.x[i] < self.zone[i][0]:
                self.x[i] = self.zone[i][0]
                self.v[i] = - self.v[i]
        for i in range(len(self.v)):
            history_influence = self.c1 * random.random() * (self.x_best[i] - self.x[i])
            peer_influence = self.c2 * random.random() * (g[i] - self.x[i])
            tabu_influence = 0
            for tabu in tabu_list:
                if tabu:
                    tabu_influence += self.c_tabu * random.random() * (self.x[i] - tabu[i])
            self.v[i] = 0.5 * self.v[i] + history_influence + peer_influence + tabu_influence


def tswarm(target_func, dimension, zone, c1=1, c2=2, c_tabu=0.35, particle_num=20, maxiter=200, verbose=False,
           visualize=False):
    swarm = []
    for _ in range(particle_num):
        swarm.append(Particle(dimension, zone, c1, c2, c_tabu))
    global_best = np.inf
    g = []
    searching_history = [([], np.inf)]
    tabu_list = [[]] * 5
    for i in range(maxiter):
        for p in swarm:
            p.get_fitness(target_func)
            if p.err_best < global_best:
                global_best = p.err_best
                g = p.x_best.copy()
        for p in swarm:
            p.update(g, tabu_list)
        if verbose:
            print(g, global_best)
        tabu_list.pop(0)
        tabu_list.append(g.copy())
        searching_history.append((g.copy(), global_best))
    if visualize:
        global_best_history = [x[1] for x in searching_history]
        sns.set_style(style='white', rc={'font.sans-serif': ['FangSong', 'simhei']})
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'font.size': 10})
        plt.xlim(0, len(searching_history))
        plt.xlabel('迭代次数')
        plt.ylabel('历史最优函数值')
        plt.plot(np.arange(1, len(searching_history)), np.array(global_best_history[1:]), color='k')
        plt.show()
    return min(searching_history, key=lambda x: x[1])


if __name__ == '__main__':
    def sphere(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)


    print(tswarm(sphere, 2, [(-10, 10), (-10, 10)], visualize=True))
