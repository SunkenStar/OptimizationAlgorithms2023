from opti_algo.tswarm import TSwarm
from tools.functions import ackley32
from tools.functions import function_index32 as func32
from tools.algo_evaluate_tool import algo_test_single


tswarm_algo = TSwarm(
    ackley32.function_body,
    ackley32.dimension,
    ackley32.zone,
    args=(1, 3, 0.4, 100, 8000),
)
result = tswarm_algo.optimize()
print(result)
tswarm_algo.visualize()

algo_test_single(
    TSwarm, func32, arguments=(1, 3, 0.4, 100, 8000), margin=0.1, verbose=True
)
