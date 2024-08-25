# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

####################################
############# PART A ###############
####################################
def BlackScholes_EuCall(t: float, S_t: float, r: float, sigma: float, T: int, K: float):
    """
    Calculate the fair price of a European Call option using the Black-Scholes formula.

    Args:
        t (int): Current time (0 ≤ t ≤ T)
        S_t (float): Stock price at time t
        r (float): Risk-free interest rate (annual rate, continuously compounded)
        sigma (float): Volatility of the stock price (annualized)
        T (float): Time to maturity of the option (T - t)
        K (float): Strike price of the option

    Returns:
        float: Fair price of the European Call option
    """
    # parameter calculation from h.w sheet
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C_t = S_t * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_t


def ImpVol(V0: float, S0: float, r: float, T: int, K: float):
    """Calculate the implied volatility of a Black-Scholes European call option.

    This function computes the implied volatility of a European call option using
    the Black-Scholes formula. Implied volatility is the volatility value that
    matches the observed market price of the option.

    Args:
        V0 (float): The observed market price of the European call option.
        S0 (float): The current price of the underlying asset.
        r (float): The risk-free interest rate, expressed as a decimal (e.g., 0.05 for 5%).
        T (int): The time to maturity of the option in years.
        K (float): The strike price of the option.

    Returns:
        float: The implied volatility of the option, expressed as a decimal (e.g., 0.2 for 20%).
    """
    # let fun be the squared difference between the BS EUCall and given V0. Our sigma is the one which minimizes this objective function
    # absolute error would yield a difficult optimisation problem so squared errors are fine
    fun = lambda sigma, V0, S0, r, T, K: (BlackScholes_EuCall(0, S0, r, sigma, T, K) - V0) ** 2
    initial_cond = 0.2
    bounds = [(0.01, 5)] # volatility is [0, infinity), for bounds lets start with a small value like 1% - (low vol regime) and max value like 500% (high vol regime like financial crisis) 
    res = minimize(fun=fun, x0=initial_cond, bounds=bounds, args=(V0, S0, r, T, K))
    return res.x[0]

# parameters
S0 = 100
r  = 0.05
T  = 1
K  = 100
V0  = 6.09

# cal of sigma
sigma = ImpVol(V0, S0, r, T, K)
print(f"Implied Volatility for part (a): {sigma}")

####################################
############# PART B ###############
####################################
def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
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

S0_range  = range(50, 151, 1)  # Range of initial stock prices
r         = 0.05
gam0      = 0.32 
kappa     = 0.32 
lamb      = 2.5
sig_tild  = 0.2
T         = 1
K         = 100
R         = 3 

# running hestom model to get V0 and then using V0 to find the sigma that minimizes error between V0 from Hestom EU call and BS EU Call
heston_eu_call_imp_vol = [ImpVol_Heston(0, S0, r, gam0, kappa, lamb, sig_tild, T, K, R) for S0 in S0_range]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S0_range, heston_eu_call_imp_vol, marker='o', linestyle='-', color='b', label="Implied Volatility Heston EU Call")
plt.xlabel("Initial Stock Price (S0)")
plt.ylabel("Implied Volatility")
plt.axvline(x=K, label="Strike Price (K)", color="red", linestyle='--')
plt.title("Heston Model Implied Volatility vs Initial Stock Price")
plt.legend()
plt.grid(True)
plt.show()


####### VOLATILITY SMILE :)))))) #######