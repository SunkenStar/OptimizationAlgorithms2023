import random
import numpy as np
from opti_algo.skeleton import Optimizer
import multiprocess as mp


class Bee:
    """
    蜜蜂类。\n
    参数说明：\n
    dimension：问题维度，或者说自变量个数\n
    zone：长度等于dimension的列表，列表中每个元素为对应位置自变量的取值范围\n
    target_func：目标函数
    """

    def __init__(self, dimension, zone, target_func):
        self.zone = zone
        self.x = []
        self.dimension = dimension
        for i in range(dimension):
            self.x.append(random.uniform(zone[i][0], zone[i][1]))
        self.err = np.inf
        self.err = target_func(self.x)
        self.k_ngh = 1
        self.exhausted = 0

    def get_neighbor_zone(self, ngh):
        """
        获取一只蜜蜂的邻域
        """
        neighbor_zone = []
        for i in range(self.dimension):
            zonesize = self.k_ngh * ngh * (self.zone[i][1] - self.zone[i][0])
            low = max(self.x[i] - zonesize, self.zone[i][0])
            high = min(self.x[i] + zonesize, self.zone[i][1])
            neighbor_zone.append((low, high))
        return neighbor_zone


class BeeAlgorithm(Optimizer):
    """
    蜜蜂算法，求取目标最小值
    """

    def __str__(self):
        return "蜜蜂算法"

    def send_workers(self, scout, ngh, workernum):
        """
        向侦查蜜蜂所在位置指定邻域发放工蜂，返回最优的一只工蜂
        """
        workers = []
        flower = scout.get_neighbor_zone(ngh)
        workers = list(
            map(
                lambda x: Bee(self.dimension, flower, self.target_func),
                range(workernum),
            )
        )
        result = min(workers, key=lambda x: x.err)
        if result.err > scout.err:
            result = scout
            result.k_ngh = scout.k_ngh * 0.8
            if result.k_ngh < 0.8 ** 3:
                result.exhausted = 1
            result.zone = self.zone
        return result

    def optimize(self):
        if self.args:
            arguments = self.args
        else:
            arguments = (66, 5, 2, 0.14, 47, 19, 50)
        n, m, e, ngh, n1, n2, max_iter = arguments
        colony = []
        for i in range(n):
            colony.append(Bee(self.dimension, self.zone, self.target_func))
        for _ in range(max_iter):
            colony.sort(key=lambda x: (x.err, x.exhausted))
            self.searching_history.append((colony[0].x, colony[0].err))
            high_fitness = colony[0:e]
            medium_fitness = colony[e:m]
            next_generation_high = list(
                map(lambda x: self.send_workers(x, ngh, n1), high_fitness)
            )
            next_generation_medium = list(
                map(lambda x: self.send_workers(x, ngh, n2), medium_fitness)
            )
            next_generation_low = list(
                map(
                    lambda x: Bee(self.dimension, self.zone, self.target_func),
                    range(n - m),
                )
            )
            colony = next_generation_high + next_generation_medium + next_generation_low
        last = min(colony, key=lambda x: x.err)
        self.searching_history.append((last.x, last.err))
        return min(self.searching_history, key=lambda x: x[1])


class ParallelBee(BeeAlgorithm):
    def __str__(self):
        return "并行蜜蜂算法"

    def send_workers(self, scout, ngh, workernum):
        """
        向侦查蜜蜂所在位置指定邻域发放工蜂，返回最优的一只工蜂
        """
        workers = []
        flower = scout.get_neighbor_zone(ngh)
        pool = mp.Pool()
        workers = pool.map(
            lambda x: Bee(self.dimension, flower, self.target_func),
            range(workernum),
        )
        pool.close()
        pool.join()
        result = min(workers, key=lambda x: x.err)
        result.zone = self.zone
        return result


if __name__ == "__main__":

    def sphere(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    bee_algo = BeeAlgorithm(sphere, 2, [(-10, 10), (-10, 10)])
    print(bee_algo.optimize())
    bee_algo.visualize()
