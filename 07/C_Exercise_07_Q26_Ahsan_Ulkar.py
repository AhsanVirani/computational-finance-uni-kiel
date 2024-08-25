# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import cmath
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T):
    ## Compute characteristic function of log-stock price in the Heston model, cf. equation (4.8) with t = 0
    d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
    phi = np.cosh(0.5 * d * T)
    psi = np.sinh(0.5 * d * T) / d
    first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi))**(2 * kappa / sig_tild ** 2)
    second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
    return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

def Heston_PCall_Lapl(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):
    # integrate w.r.t u
    def integrand(u):
        z1 = complex(R,u)     # R + iu
        z2 = complex(u, -R)   # u - iR 
        # eq 4.6 - our lower limit now logK/p
        f_tilde_z = - (K ** ((p - z1) / p) / (p - z1)) - (K ** (1- z1 / p) / (z1))
        # pg 52 first equation
        characteristic_eq = heston_char(z2, S0, r, gam0, kappa, lamb, sig_tild, T)
        # eq 4.4
        return (np.exp(-r * T) / np.pi) * np.real(f_tilde_z * characteristic_eq)

    # Perform integration
    I = integrate.quad(integrand, 0, np.inf)[0]

    # Return the option price
    return I

# setting variables
power_call_price = []

S0_range  = range(50, 151, 1)  # Range of initial stock prices
r         = 0.05
gam0      = 0.32 
kappa     = 0.32 
lamb      = 2.5
sig_tild  = 0.2
T         = 1
K         = 100
p         = 1
R         = p + 1  # greater than p

# Compute option price for each S0
list(map(lambda S0: power_call_price.append(Heston_PCall_Lapl(S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p)), S0_range))

plt.plot(S0_range, power_call_price, 'r', label = 'Power Call Option Valuation')
plt.axvline(x=K, color='b', linestyle='--', label='Strike Price (K)')
plt.xlabel('Stock Price (S0)')
plt.ylabel('Power Call Price')
plt.legend()
plt.show()