from numpy import abs
from tswarm import tswarm
from functions import salomon

pso_fail = 0
tabu_fail = 0

for _ in range(100):
    res = tswarm(salomon.function_body, salomon.dimension, salomon.zone, c_tabu=0)
    delta = abs(salomon.best - res[1])
    if delta > 1e-3:
        pso_fail += 1

for _ in range(100):
    res = tswarm(salomon.function_body, salomon.dimension, salomon.zone)
    delta = abs(salomon.best - res[1])
    if delta > 1e-3:
        tabu_fail += 1
        
print(f'PSO: {pso_fail}% Fail')
print(f'TSwarm: {tabu_fail}% Fail')
