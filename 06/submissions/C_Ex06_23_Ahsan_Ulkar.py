# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import cmath
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def BS_EuCall_Laplace(S0: float, r: float, sigma: float, T: int, K: float, R: int):
    # integrate w.r.t u
    def integrand(u):
        z1 = complex(R,u)     # R + iu
        z2 = complex(u, -R)   # u - iR 
        z3 = complex(0, 1)
        # eq 4.6
        f_tilde_z = (K ** (1 - z1)) / (z1 * (z1 - 1))
        # pg 52 first equation
        characteristic_eq = cmath.exp(z3 * z2 * (np.log(S0) + r * T) - (z3 * z2 + z2 ** 2) * (sigma ** 2 * T / 2))
        # eq 4.4
        return (np.exp(-r * T) / np.pi) * np.real(f_tilde_z * characteristic_eq)

    # Perform integration
    I = integrate.quad(integrand, 0, np.inf)[0]

    # Return the option price
    return I

# setting variables
EU_Call_price = []

S0_range = range(50, 151, 1)  # Range of initial stock prices
r = 0.03
sigma = 0.2
T = 1
K = 110
R = 10  # Chosen positive constant for convergence. R > 1 for call option.

# Compute option price for each S0
list(map(lambda S0: EU_Call_price.append(BS_EuCall_Laplace(S0, r, sigma, T, K, R)), S0_range))

plt.plot(S0_range, EU_Call_price, 'r', label = 'EU Call Option Valuation')
plt.axvline(x=K, color='b', linestyle='--', label='Strike Price (K)')
plt.xlabel('Stock Price (S0)')
plt.ylabel('EU Call Price')
plt.legend()
plt.show()
