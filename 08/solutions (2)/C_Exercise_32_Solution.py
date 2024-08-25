import numpy as np
import math
import scipy.stats as ss


def DigitalOptionHedge_BS_MC_LR(S,r, sigma, T, t, K, m):
    # sample X under P_s
    X = np.random.normal(math.log(S)/math.sqrt(sigma**2*T), 1, m)
    ST = np.exp((r - sigma ** 2 / 2) * T + sigma * math.sqrt(T) * X)

    # follow p.63
    f = lambda ST: math.exp(-r * T) * np.where(ST > K, 1, 0)
    d_log_g = lambda x: x/(S*math.sqrt(sigma**2*T)) - math.log(S)/(S*sigma**2*T)
    Z = f(ST) * d_log_g(X)
    phi_1 = np.mean(Z)
    return phi_1


# add the closed-form solution from T18 as comparison
def Digital_Call_Hedge(S, r, sigma, T, t, K):
    d2 = (math.log(S/K) + (r- sigma**2/2)*(T-t))/(sigma * math.sqrt(T-t))
    phi_1 = math.exp(-r*(T-t)) * ss.norm.pdf(d2) * 1/(S*sigma*math.sqrt(T-t))
    return phi_1

S0 = 100
r = 0.05
sigma = 0.2
T = 1
t = 0
K = 100
m = 100000

print(str(Digital_Call_Hedge(S0, r, sigma, T, t, K)))
print(str(DigitalOptionHedge_BS_MC_LR(S0, r, sigma, T, t, K, m)))