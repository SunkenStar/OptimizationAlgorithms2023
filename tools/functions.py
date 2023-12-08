"""
智能优化算法系列测试函数，为2023秋计算智能算法课程期末作业设计。\n
function_index收录十种来自infinity77.net的测试函数\n
建议配合algo_evaluate_tool使用\n
"""

import numpy as np


class TestFunction:
    """
    测试函数类
    """

    def __init__(self, dimension, zone, best_location, best, name):
        self.dimension = dimension
        self.zone = zone
        self.best_location = best_location
        self.best = best
        self.name = name

    @staticmethod
    def function_body(x):
        """
        函数体
        """


class Ackley(TestFunction):
    @staticmethod
    def function_body(x):
        a = -20 * np.e ** (-0.2 * np.sqrt(sum([k ** 2 for k in x]) / len(x)))
        b = -np.e ** ((sum([np.cos(2 * np.pi * k) for k in x])) / len(x))
        return a + b + 20 + np.e


ackley = Ackley(2, [(-32, 32), (-32, 32)],
                (0, 0), 0, 'ackley')


class Alpine01(TestFunction):
    @staticmethod
    def function_body(x):
        variable = np.array(x)
        return np.sum(np.abs(np.multiply(variable, np.sin(variable)) + 0.1 * variable))


alpine01 = Alpine01(2, [(-10, 10), (-10, 10)],
                    (0, 0), 0, 'alpine01')


class Bartelsconn(TestFunction):
    @staticmethod
    def function_body(x):
        a = np.abs(x[0] ** 2 + x[1] ** 2 + x[0] * x[1])
        return a + np.abs(np.sin(x[0])) + np.abs(np.cos(x[1]))


bartelsconn = Bartelsconn(2, [(-50, 50), (-50, 50)],
                          (0, 0), 1, 'bartelsconn')


class Bohachevsky(TestFunction):
    @staticmethod
    def function_body(x):
        a = x[0] ** 2 + 2 * x[1] ** 2
        b = -0.3 * np.cos(3 * np.pi * x[0])
        c = -0.4 * np.cos(4 * np.pi * x[1])
        return a + b + c + 0.7


bohachevsky = Bohachevsky(2, [(-15, 15), (-15, 15)],
                          (0, 0), 0, 'bohachevsky')


class Rastrigin(TestFunction):
    @staticmethod
    def function_body(x):
        return 20 + x[0] ** 2 + x[1] ** 2 - 10 * np.cos(2 * np.pi * x[0]) \
            - 10 * np.cos(2 * np.pi * x[1])


rastrigin = Rastrigin(2, [(-4.5, 4.5), (-4.5, 4.5)],
                      (0, 0), 0, 'rastrigin')


class Eggcrate(TestFunction):
    @staticmethod
    def function_body(x):
        a = x[0] ** 2 + x[1] ** 2
        b = 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)
        return a + b


eggcrate = Eggcrate(2, [(-5, 5), (-5, 5)],
                    (0, 0), 0, 'eggcrate')


class Exponential(TestFunction):
    @staticmethod
    def function_body(x):
        return -1 * np.e ** (-0.5 * sum([k ** 2 for k in x]))


exponential = Exponential(2, [(-1, 1), (-1, 1)],
                          (0, 0), -1, 'exponential')


class Salomon(TestFunction):
    @staticmethod
    def function_body(x):
        a = np.cos(2 * np.pi * np.sqrt(sum([k ** 2 for k in x])))
        b = 0.1 * np.sqrt(sum([k ** 2 for k in x]))
        return 1 - a + b


salomon = Salomon(2, [(-100, 100), (-100, 100)],
                  (0, 0), 0, 'salomon')


class Himmerblau(TestFunction):
    @staticmethod
    def function_body(x):
        a = (x[0] ** 2 + x[1] - 11) ** 2
        b = (x[0] + x[1] ** 2 - 7) ** 2
        return a + b


himmerblau = Himmerblau(2, [(-6, 6), (-6, 6)],
                        (0, 0), 0, 'himmerblau')


class MultiModal(TestFunction):
    @staticmethod
    def function_body(x):
        a = sum([np.abs(k) for k in x])
        b = np.prod(np.array([np.abs(k) for k in x]))
        return a * b


multimodal = MultiModal(2, [(-10, 10), (10, 10)],
                        (0, 0), 0, 'multimodal')

function_index = [ackley, alpine01, bartelsconn, bohachevsky,
                  eggcrate, exponential, salomon, himmerblau,
                  multimodal, rastrigin]
