from opti_algo.tswarm import tswarm
from opti_algo.bee_algo import bee_algo
from tools.algo_evaluate_tool import algo_test_single, algo_test_multi
from tools.functions import function_index as funcs


algo_test_single(tswarm, '禁忌粒子群算法', funcs)
algo_test_single(bee_algo, '蜜蜂算法', funcs)
algo_test_multi(tswarm, '禁忌粒子群算法', funcs)
algo_test_multi(bee_algo, '蜜蜂算法', funcs)
