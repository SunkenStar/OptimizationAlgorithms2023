import numpy as np
import multiprocessing as mp
from opti_algo.bee_algo import Bee


def send_workers(order):
    scout, ngh, dimension, target_func, workernum, dst = order
    workers = []
    for _ in range(workernum):
        workers.append(Bee(dimension, scout.get_neighbor_zone(ngh), target_func))
    return min(workers, key=lambda x: x.err)


def parallel_bee(target_func, dimension, zone, n=100, m=24, e=6, ngh=0.5, n1=40, n2=20, max_iter=10):
    colony = []
    pool = mp.Pool(4)
    for i in range(n):
        colony.append(Bee(dimension, zone, target_func))
    for _ in range(max_iter):
        colony.sort(key=lambda x: x.err)
        high_fitness = colony[0:e]
        medium_fitness = colony[e:m]
        next_generation = []
        orders_high = [(scout, ngh, dimension, target_func, n1, next_generation) for scout in high_fitness]
        orders_medium = [(scout, ngh, dimension, target_func, n2, next_generation) for scout in medium_fitness]
        orders = orders_high + orders_medium
        next_generation = pool.map(send_workers, orders)
        while len(next_generation) < n:
            next_generation.append(Bee(dimension, zone, target_func))
        colony = next_generation.copy()
        ngh *= 0.8
    result = min(colony, key=lambda x: x.err)
    pool.close()
    pool.join()
    return result.x, result.err


if __name__ == '__main__':
    def sphere(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)


    print(parallel_bee(sphere, 2, [(-10, 10), (-10, 10)]))
    