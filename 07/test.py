import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

def BlackScholes_EuCall(t: float, S_t: float, r: float, sigma: float, T: int, K: float):
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C_t = S_t * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_t

def ImpVol(V0: float, S0: float, r: float, T: int, K: float):
    fun = lambda sigma, V0, S0, r, T, K: (BlackScholes_EuCall(0, S0, r, sigma, T, K) - V0) ** 2
    initial_cond = 0.2
    res = minimize(fun=fun, x0=initial_cond, args=(V0, S0, r, T, K))
    return res.x[0]

S0 = 100
r  = 0.05
T  = 1
K  = 100
V0  = 6.09

sigma = ImpVol(V0, S0, r, T, K)
print(f"Implied Volatility for part (a): {sigma}")

def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

def Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
    K = np.atleast_1d(K)
    f_tilde_0 = lambda u: 1 / (u * (u - 1))
    chi_0 = lambda u: heston_char(u, S0=S0, r=r, gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=T)
    g = lambda u: f_tilde_0(R + 1j * u) * chi_0(u - 1j * R)

    kappa_1 = np.log(K[0])
    M = np.minimum(2 * np.pi * (N - 1) / (np.log(K[-1] - kappa_1)), 500)
    Delta = M / N
    n = np.arange(1, N + 1)
    kappa_m = np.linspace(kappa_1, kappa_1 + 2 * np.pi * (N - 1) / M, N)

    x = g((n - 0.5) * Delta) * Delta * np.exp(-1j * (n - 1) * Delta * kappa_1)
    x_hat = np.fft.fft(x)

    V_kappa_m = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-0.5 * Delta * kappa_m * 1j))
    return interp1d(kappa_m, V_kappa_m)(np.log(K))

def ImpVol_Heston(V0, S0, r, gam0, kappa, lamb, sig_tild, T, K, R):
    V0 = Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, 2 ** 15)
    return ImpVol(V0, S0, r, T, K)

r = 0.05
T = 1
K = 100
gam0 = 0.05
kappa = 0.5
lamb = 0.5
sig_tild = 0.1
R = 0.1
S0 = np.arange(60, 145, 5)
IV_heston = [ImpVol_Heston(0, s, r, gam0, kappa, lamb, sig_tild, T, K, R) for s in S0]

plt.plot(S0, IV_heston)
plt.xlabel('Stock Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Stock Price for Heston Model')
plt.show()
