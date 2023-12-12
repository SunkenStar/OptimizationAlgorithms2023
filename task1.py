from opti_algo.tswarm import TSwarm
from opti_algo.bee_algo import BeeAlgorithm
from opti_algo.bee_algo import ParallelBee
from tools.algo_evaluate_tool import algo_test_single, algo_test_multi
from tools.functions import function_index as funcs


if __name__ == "__main__":
    algo_test_multi(TSwarm, funcs, times=50, arguments=(1, 3, 0.4, 10, 50))
    # algo_test_multi(BeeAlgorithm, funcs, times=50)
    # algo_test_multi(ParallelBee, funcs, times=50)


# def wrapped_bee(x):
#     t = x.copy()
#     for i in (0,1,2,5,6):
#         t[i]=int(x[i])
#     t.append(50)
#     return 1/algo_test_multi(BeeAlgorithm, funcs, arguments=t, times=10, silent=True)[0]

# tswarm_algo = TSwarm(
#     wrapped_bee,
#     dimension=7,
#     zone=[(30,60),(4,6),(1,4),(0.5,0.8),(0.5,0.9),(30,50),(10,20)],
#     args=(1, 3, 0.4, 20, 100),
# )
# result = tswarm_algo.optimize()
# print(result)
