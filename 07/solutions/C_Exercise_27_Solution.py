import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.optimize import minimize

from InTutorialExercise13 import Heston_EuCall


# computes the price of a put in the Black-Scholes model
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    d_1 = (np.log(S_t / K) + (r + 1 / 2 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d_2 = d_1 - sigma * np.sqrt(T - t)
    phi = ss.norm.cdf(d_1)
    C = S_t * phi - K * np.exp(-r * (T - t)) * ss.norm.cdf(d_2)
    return C


# part (a)
def ImpVol(V0, S0, r, T, K):
    def function(sigma):
        squared_error = ((V0 - BlackScholes_EuCall(0, S0, r, sigma, T, K)) ** 2).mean()
        return squared_error
    initial_guess = 0.3
    res = minimize(function, initial_guess, bounds=((0, None),), method='Powell')
    return res.x[0]

# Test it with a Blackscholes price:
S0 = 100
r = 0.05
T = 1
K = 100
V0 = 10
print("The implied volatility is " + str(ImpVol(V0, S0, r, T, K)))


# part (b)
def ImpVol_Heston(S0, r, gam0, kappa, lamb, sig_tild, T, K, R):
    V0 = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N=2**15)
    print(V0)
    implVol = np.empty(len(K), dtype=float)
    for i in range(0, len(K)):
        implVol[i] = ImpVol(V0[i], S0, r, T, K[i])
    return implVol, V0


S0 = 100
r = 0.05
gam0 = 0.3 ** 2
kappa = 0.3 ** 2
lamb = 2.5
sig_tild = 0.2
T = 1
K = np.linspace(80, 180, 101)
R = 1.5

implVol, V0 = ImpVol_Heston(S0, r, gam0, kappa, lamb, sig_tild, T, K, R)

plt.figure(figsize=(8, 6))
plt.plot(K, implVol)
plt.xlabel('Strike $K$')
plt.ylabel('Implied Volatility')
plt.grid(alpha=0.4)
plt.show()
