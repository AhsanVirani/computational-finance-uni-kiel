# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Tuple, Callable

# compute Black-Scholes price by integration - copied from the resource material
def BS_Price_Int(S0: float, r: float, sigma: float, T: int, f: Callable[[float], float]) -> float:
    # define integrand as given in the exercise

    def integrand(x):
        return 1 / math.sqrt(2 * math.pi) * f(
            S0 * math.exp((r - 0.5 * math.pow(sigma, 2)) * T + sigma * math.sqrt(T) * x)) * math.exp(-r * T) * math.exp(
            -1 / 2 * math.pow(x, 2))

    # perform integration
    I = integrate.quad(integrand, -np.inf, np.inf)
    
    # return value of the integration
    return I[0]

def BS_Greeks_num(r: float, sigma: float, S0: float, T: int, g: Callable[[float], float], eps: float) -> Tuple[float, float, float]:
    """
    Computes the Greeks of an option using numerical differentiation.

    Args:
        r (float): Risk-free interest rate (must be non-negative).
        sigma (float): Volatility of the stock (must be positive).
        S0 (float): Initial price of the stock (must be positive).
        T (int): Maturity time in years (must be non-negative).
        g (Callable[[float], float]): Payoff function of the option, takes stock price as input.
        eps (float): Small precision value for numerical differentiation (must be positive).

    Returns:
        Tuple[float, float, float]: A tuple containing the Delta, Vega, and Gamma of the option in that order.

    Raises:
        ValueError: If any of the input parameters are not in their valid ranges.
    """
    V0_BS = BS_Price_Int(S0, r, sigma, T, g)
    # measure of the rate of change of option price with result to changes in the underlying asset's price - sensitivity of the option price
    delta = (BS_Price_Int(S0 + eps * S0, r, sigma, T, g) - V0_BS) / (eps * S0)
    # measures the sensitivity of the option's price to changes in the volatility of the underlying asset
    vega = (BS_Price_Int(S0, r, sigma + sigma * eps, T, g) - V0_BS) / (eps * sigma)
    # measure of the rate of change of Delta with respect to changes in the underlying asset's price.
    gamma = (BS_Price_Int(S0 + eps * S0, r, sigma, T, g) - 2 * V0_BS + BS_Price_Int(S0 - eps * S0, r, sigma, T, g)) / (eps * S0) ** 2

    return (delta, vega, gamma)


# plotting Delta of the option
r     = 0.05
sigma = 0.3
T     = 1
S0    = range(60, 141, 1)
K     = 110
g     = lambda x: np.maximum(x - K, 0)
eps   = 0.001

delta = np.zeros(len(S0))

for idx, val in enumerate(S0):
    delta[idx] = BS_Greeks_num(S0=val,r=r, sigma=sigma, T=T, g=g, eps=eps)[0] # only need delta

plt.plot(S0, delta, 'r', label="Delta against Stock Price")
plt.title("Delta of European Call Option in Black-Scholes Model")
plt.axvline(x=K, color='b', linestyle='--', label='Strike Price (K)')
plt.ylabel('Delta of the EU_Call Option')
plt.xlabel('Stock Price (S0)')
plt.legend()
plt.show()

