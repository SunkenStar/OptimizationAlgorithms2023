from numpy import abs
from tswarm import tswarm
from functions import function_index

for func in function_index:
    print(f'{func.name}函数优化结果')
    res = tswarm(func.function_body,
                 func.dimension,
                 func.zone,
                 visualize=False)
    print(res)
    print('理论值')
    print(func.best_location, func.best)
    delta = abs(func.best - res[1])
    print(f'误差{delta}')
    if delta > 1e-3:
        print(f'优化失败！')
    print('\n')
    