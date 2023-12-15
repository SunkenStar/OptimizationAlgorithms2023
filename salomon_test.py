from numpy import abs
from opti_algo.tswarm import TSwarm
from tools.functions import salomon, salomon32

pso_fail = 0
tabu_fail = 0

print("salomon函数测试")

for _ in range(100):
    pso = TSwarm(
        salomon.function_body, salomon.dimension, salomon.zone, args=(1, 3, 0, 40, 200)
    )
    res = pso.optimize()
    delta = abs(salomon.best - res[1])
    if delta > 1e-3:
        pso_fail += 1

for _ in range(100):
    tswarm_algo = TSwarm(salomon.function_body, salomon.dimension, salomon.zone)
    res = tswarm_algo.optimize()
    delta = abs(salomon.best - res[1])
    if delta > 1e-3:
        tabu_fail += 1

print(f"PSO: {pso_fail}% Fail")
print(f"TSwarm: {tabu_fail}% Fail")

print("\n")

print("salomon函数32维测试")

for _ in range(10):
    pso = TSwarm(
        salomon32.function_body,
        salomon32.dimension,
        salomon32.zone,
        (1, 3, 0, 100, 8000),
    )
    res = pso.optimize()
    delta = abs(salomon.best - res[1])
    if delta > 1e-1:
        pso_fail += 1

for _ in range(10):
    tswarm_algo = TSwarm(
        salomon32.function_body,
        salomon32.dimension,
        salomon32.zone,
        (1, 3, 0.4, 100, 8000),
    )
    res = tswarm_algo.optimize()
    delta = abs(salomon.best - res[1])
    if delta > 1e-1:
        tabu_fail += 1

print(f"PSO: {pso_fail}% Fail")
print(f"TSwarm: {tabu_fail}% Fail")
