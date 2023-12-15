from opti_algo.bee_algo import BeeAlgorithm
from opti_algo.tswarm import TSwarm
import control as ct
import matplotlib.pyplot as plt
import numpy as np

s = ct.tf([1, 0], [1])
K_amp = 0.05
G_valve = 3.06 * 10**-3 / (s**2 / 600**2 + 2 * 0.5 * s / 600 + 1)
G_motor = 1.25 * 10**6 / (s**2 / 388**2 + 2 * 0.94 * s / 388 + 1)
K_gear = 3
K_sensor = 0.175
G = K_amp * G_valve * G_motor * K_gear * K_sensor

def compensatorPID(param):
    a, b = param
    Gc = ct.tf([a, 1], [1]) * ct.tf([b, 1], [1, 0])
    G_hat = Gc * G
    gm, pm, wcg, wcp = ct.margin(G_hat)
    if pm < 40 or pm > 60 or wcp > 1000:
        return 1e10
    return np.abs(pm+140) + np.abs(wcp-1000)
    

def test_resultPID(param):
    a, b = param
    Gc = ct.tf([a, 1], [1]) * ct.tf([b, 1], [1, 0])
    G_hat = Gc * G
    ct.bode(G_hat, omega = np.logspace(start = 0, stop = 4, num = 200), dB=True, deg=True)
    plt.show()
    # ct.nyquist_plot(G_hat)
    # plt.show()
    return ct.margin(G_hat)

def compensator(param):
    a, b, T1, T2 = param
    Gc = ct.tf([a*T1, 1], [T1, 1]) * ct.tf([b*T2, 1], [T2, 0])
    G_hat = Gc * G / s
    G_hat = Gc * G
    gm, pm, wcg, wcp = ct.margin(G_hat)
    mag1, phase1, omega1 = ct.freqresp(G_hat, 10)
    mag2, phase2, omega2 = ct.freqresp(G_hat, 10)
    if pm < 40 or pm > 60 or wcp > 1000 or mag1 < 15 or mag2 < 5:
        return 1e10
    return np.abs(pm+140) + np.abs(wcp-1000) 
    

def test_result(param):
    a, b, T1, T2 = param
    Gc = ct.tf([a*T1, 1], [T1, 1]) * ct.tf([b*T2, 1], [T2, 0])
    G_hat = Gc * G / s
    G_hat = Gc * G
    ct.bode(G_hat, omega = np.logspace(start = 0, stop = 4, num = 200), dB=True, deg=True)
    plt.show()
    # ct.nyquist_plot(G_hat)
    # plt.show()
    return ct.margin(G_hat)

guess = [(1,10),(0,1),(0,1),(1,10)]

bee_algo = BeeAlgorithm(compensator, 4, guess, (800, 100, 50, 0.6, 0.8, 80, 40, 80))
res = bee_algo.optimize()
print(res)
bee_algo.visualize()
print(test_result(res[0]))

# tswarm = TSwarm(compensator, 2, guess)
# res = tswarm.optimize()
# print(res)
# tswarm.visualize()
# print(test_result(res[0]))