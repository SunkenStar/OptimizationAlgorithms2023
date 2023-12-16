from opti_algo.tswarm import TSwarm
from opti_algo.bee_algo import BeeAlgorithm
from opti_algo.bee_algo import ParallelBee
from tools.algo_evaluate_tool import algo_test_single, algo_test_multi
from tools.functions import function_index as funcs
from tools.functions import function_index32 as func32
from tools.functions import ackley, ackley32


if __name__ == "__main__":
    algo_test_single(TSwarm, funcs, verbose=True)
    algo_test_single(BeeAlgorithm, funcs, verbose=True)
    algo_test_multi(TSwarm, funcs, times=50)
    algo_test_multi(BeeAlgorithm, funcs, times=50)
    algo_test_single(
        TSwarm, func32, arguments=(1, 3, 0.4, 100, 8000), margin=0.1, verbose=True
    )
    algo_test_single(
        BeeAlgorithm, func32, arguments=(200, 25, 10, 0.5, 140, 60, 250), margin=0.1, verbose=True
    )
    bee_algo = BeeAlgorithm(
        ackley.function_body,
        ackley.dimension,
        ackley.zone,
    )
    result = bee_algo.optimize()
    print(result)
    bee_algo.visualize()
    tswarm_algo = TSwarm(
        ackley32.function_body,
        ackley32.dimension,
        ackley32.zone,
        args=(1, 3, 0.4, 100, 8000),
    )
    result = tswarm_algo.optimize()
    print(result)
    tswarm_algo.visualize()



