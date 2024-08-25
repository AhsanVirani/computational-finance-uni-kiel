# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import numpy as np
import scipy


def EuOptionHedge_BS_MC_IP(St, r, sigma, g, T, t, N):
    # empty list to store Z
    Z = []
    # generate N standard normal variables
    X = standard_normals = np.random.randn(N)
    for i in range(0, len(X)):
        # simplified on pg 62 eq
        ST = St * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i]) 
        indicator_logical = g(ST) > 0
        exponential_term  = np.exp(-sigma ** 2 * (T - t) / 2 + sigma * np.sqrt(T - t) * X[i])
        Z.append(indicator_logical * exponential_term)

    Z_prime = np.mean(Z)
    return Z_prime
    
        
# Ex. 3.30
def BS_EU_Call_Delta(St, r, sigma, t, T, K):
    d1 = (np.log(St / K) + r * (T - t) + (sigma ** 2 / 2) * (T - t)) / (sigma * np.sqrt(T - t))
    return scipy.stats.norm.cdf(d1)


# init parameters for testing
t     = 0
S0    = 100
r     = 0.05
sigma = 0.2
T     = 1
N     = 10000
K     = 90
g_x   = lambda x: np.maximum(x - K, 0)


IP_hedge = EuOptionHedge_BS_MC_IP(S0, r, sigma, g_x, T, t, N)
BS_EU_Call_Hedge = BS_EU_Call_Delta(S0, r, sigma, t, T, K)

print(f'Hedge with infinitesimal perturbation: {IP_hedge} \nHedge with BS implementation: {BS_EU_Call_Hedge}')