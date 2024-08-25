# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Loading data
data = pd.read_csv("07/option_prices_sp500.csv")

T = 1  # 1 year maturity - given in the Question

# taking from Q 27
def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / np.where(d == 0, 1, d)  # Handle division by zero

    first_factor = np.exp(lamb * T / 2) / (phi + lamb * psi)
    first_factor = np.where(np.isfinite(first_factor), first_factor, 0)  # Replace non-finite values with 0
    first_factor = first_factor ** (2 * kappa / sig_tild ** 2)

    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    second_factor = np.where(np.isfinite(second_factor), second_factor, 0)  # Replace non-finite values with 0

    result = np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor
    result = np.where(np.isfinite(result), result, 0)  # Replace non-finite values with 0

    return result

# taking from Q 27
def Heston_EuCall(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
    K = np.atleast_1d(K)
    f_tilde_0 = lambda u: 1 / (u * (u - 1))
    chi_0 = lambda u: heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T)
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

# Objective function as defined - note that we take N = 2 ** 15 as given in Tut 13
def objective_function(param, V0_data, S0, K, r, T):
    kappa, lamb, sig_tild, gam0 = param
    squared_errors = 0
    for i in range(len(V0_data)):
        squared_errors += (Heston_EuCall(S0.iloc[i], r.iloc[i], gam0, kappa, lamb, sig_tild, T, K.iloc[i], 3, 2 ** 15) - V0_data.iloc[i]) ** 2
    return squared_errors

def calibrate_Heston(V0_data, S0, r, K, T):
    """
    Calibrate the Heston model parameters by minimizing the sum of squared errors between
    the model-implied option prices and the observed option prices.

    Args:
        V0_data (pandas.Series): Observed option prices from the dataset.
        S0 (pandas.Series): Underlying asset prices.
        r (pandas.Series): Interest rates.
        K (pandas.Series): Strike prices.
        T (float): Time to maturity.

    Returns:
        tuple: Calibrated Heston model parameters (kappa, lamb, sig_tild, gam0).

    """
    initial_cond = [1.5, 0.5, 0.2, 0.1]  # kappa, lamb, sig_tild, gam0
    bounds = [(0.01, 5), (0.01, 3), (0.01, 2), (0.01, 0.5)]  # Lower and upper bounds - all params [0, infinity) we take reasonable bounds for convergence after testing a few
    res = minimize(fun=objective_function, x0=initial_cond, args=(V0_data, S0, K, r, T), method='L-BFGS-B', bounds=bounds)
    # kappa, lamb, sig_tild, gam0
    return res.x

# setting parameters - note these are series objects
V0_data = data['OptionPrice']
S0      = data['Underlying']
r       = data['InterestRate']
K       = data['Strike']

kappa, lamb, sig_tild, gam0 = calibrate_Heston(V0_data, S0, r, K, T)
print("Calibrated Parameters:", kappa, lamb, sig_tild, gam0)

# computing Heston_EuCall using calibrated params for different strike prices K - note that underlying S0 and r are the same over the whole series
model_prices = [Heston_EuCall(S0.iloc[i], r.iloc[i], gam0, kappa, lamb, sig_tild, T, K.iloc[i], 3, 2 ** 15) for i in range(len(S0))]

plt.figure(figsize=(12, 6))
plt.scatter(K, V0_data, label='Observed Option Prices', color='grey', alpha=0.5)
plt.plot(K, model_prices, label='Heston Model Prices with Calibrated Parameters')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price (V0)')
plt.title('Observed vs. Heston Model Option Prices with Calibrated Parameters')
plt.legend()
plt.show()
