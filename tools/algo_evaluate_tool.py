"""
智能优化算法评估工具包，为2023秋计算智能算法课程期末作业设计。\n
使用的测试函数需要基于functions.py中定义的TestFunction类编写\n
"""
from numpy import abs
from time import time


def algo_test_single(
    algorithm, function_index, arguments=None, margin=1e-3, verbose=False, silent=False
):
    """
    单次测试函数。使用给定算法优化给定目标函数列表中的函数，返回通过次数和总测试次数。\n
    参数说明：\n
    algorithm：使用的算法类，要求继承skeleton.py中定义的Optimizer抽象基类\n
    name：算法名，字符串\n
    function_index：存储所有目标函数的列表，各元素要求继承functions.py中定义的TestFunction类\n
    margin：允许的结果误差范围\n
    verbose：若为真，详细显示测试过程中列表中各个函数的优化情况\n
    silent：若为真，本函数将不产生任何控制台输出\n
    """
    dummy = algorithm(None, None, None)
    if not silent:
        print(f"正在测试{dummy}")
    passtest = 0
    faillist = []
    for func in function_index:
        algo_instance = algorithm(
            func.function_body, func.dimension, func.zone, arguments
        )
        res = algo_instance.optimize()
        delta = abs(func.best - res[1])
        if delta < margin:
            passtest += 1
        else:
            faillist.append((func.name, delta))
        if verbose and not silent:
            print(f"{func.name}函数优化结果")
            print(res)
            print("理论值")
            print(func.best_location, func.best)
            print(f"误差{delta}")
            if delta > margin:
                print(f"优化失败！")
            print("\n")
    if not silent:
        print(f"通过测试用例{passtest}/{len(function_index)}")
        for failure in faillist:
            print(f"其中{failure[0]}函数未通过测试，误差为{failure[1]}")
        print("\n")
    return passtest, len(function_index)


def algo_test_multi(algorithm, function_index, arguments=None, times=100, margin=1e-3, silent=False):
    """
    多次测试函数。相当于重复多次执行单次测试函数，削弱算法随机性对测试结果带来的影响。\n
    在控制台上输出总测试轮数和通过率。\n
    参数说明：\n
    algorithm：使用的算法，规定为函数形式，接收目标函数、问题维数、以[(下限,上限),(下限,上限)]格式保存的自变量各维度取值范围\n
    name：算法名，字符串\n
    function_index：存储所有目标函数的列表，各元素应当基于functions.py中定义的TestFunction类编写\n
    times：重复测试次数
    margin：允许的结果误差范围\n
    """
    dummy = algorithm(None, None, None)
    if not silent:
        print(f"开始对{dummy}的多轮测试")
    total_passes = 0
    total_tests = 0
    time_start = time()
    for _ in range(times):
        single_res = algo_test_single(
            algorithm, function_index, arguments, margin, False, True
        )
        total_passes += single_res[0]
        total_tests += single_res[1]
    time_consumed = time() - time_start
    if not silent:
        print(f"进行{total_tests}轮测试，{dummy}通过{total_passes}轮")
        print(f"通过率{total_passes / total_tests * 100:.2f}%，耗时{time_consumed:.3f}s\n")
    return total_passes / total_tests, time_consumed
