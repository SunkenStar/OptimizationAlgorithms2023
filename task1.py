from opti_algo.tswarm import TSwarm
from opti_algo.bee_algo import BeeAlgorithm
from opti_algo.bee_algo import ParallelBee
from tools.algo_evaluate_tool import algo_test_single, algo_test_multi
from tools.functions import function_index as funcs


if __name__ == "__main__":
    # algo_test_multi(TSwarm, funcs, times=50)
    # algo_test_multi(BeeAlgorithm, funcs, times=50)
#     # algo_test_multi(ParallelBee, funcs, times=50)
#    algo_test_single(TSwarm, funcs, verbose=True)
    pass


def wrapped_bee(x):
    t = x.copy()
    for i in (0,1,2,4,5):
        t[i]=int(x[i])
    t.append(50)
    testpass, total, time_consumed = algo_test_multi(BeeAlgorithm, funcs, arguments=t, times=50, silent=True)
    if time_consumed > 20 or testpass < total - 60:
        return 1e10
    return total - testpass + time_consumed

tswarm_algo = TSwarm(
    wrapped_bee,
    dimension=6,
    zone=[(30,80),(3,6),(1,3),(0,1),(30,50),(10,20)],
    args=(1, 3, 0.4, 10, 50),
)
result = tswarm_algo.optimize()
print(result)
