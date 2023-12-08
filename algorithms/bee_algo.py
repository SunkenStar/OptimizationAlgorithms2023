import random
import numpy as np


class _Bee:
    def __init__(self, dimension, zone, target_func):
        self.zone = zone
        self.x = []
        self.dimension = dimension
        for i in range(dimension):
            self.x.append(random.uniform(zone[i][0], zone[i][1]))
        self.err = np.inf
        self.err = target_func(self.x)

    def get_neighbor_zone(self, ngh):
        neighbor_zone = []
        for dim in self.x:
            neighbor_zone.append((dim - ngh, dim + ngh))
        return neighbor_zone


def bee_algo(target_func, dimension, zone, n=100, m=20, e=5, ngh=1, n1=40, n2=20, max_iter=100):
    colony = []
    for i in range(n):
        colony.append(_Bee(dimension, zone, target_func))
    for _ in range(max_iter):
        colony.sort(key=lambda x: x.err)
        high_fitness = colony[0:e]
        medium_fitness = colony[e:m]
        next_generation = []
        for scout in high_fitness:
            workers = []
            for _ in range(n1):
                workers.append(_Bee(dimension, scout.get_neighbor_zone(ngh), target_func))
            next_generation.append(min(workers, key=lambda x: x.err))
        for scout in medium_fitness:
            workers = []
            for _ in range(n2):
                workers.append(_Bee(dimension, scout.get_neighbor_zone(ngh), target_func))
            next_generation.append(min(workers, key=lambda x: x.err))
        while len(next_generation) < n:
            next_generation.append(_Bee(dimension, zone, target_func))
        colony = next_generation
        ngh *= 0.8
    result = min(colony, key=lambda x: x.err)
    return result.x, result.err


if __name__ == '__main__':
    def sphere(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)


    print(bee_algo(sphere, 2, [(-10, 10), (-10, 10)]))
