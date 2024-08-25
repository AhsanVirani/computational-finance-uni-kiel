# C-Exercise 26, SS 2024

import math
import cmath
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt


def Heston_PCall_Laplace(S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R, p):

    # Laplace transform of the function f(x) = (e^(xp) - K)^+ (cf. (4.6))
    def f_tilde(z):
        return p * cmath.exp((1 - z / p) * math.log(K)) / (z * (z - p))

    # Characteristic function of log(S(T)) in the Heston model (cf. (4.8))
    def chi(u):
        d = cmath.sqrt(
            math.pow(lmbda, 2) + math.pow(sigma_tilde, 2) * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
        n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
        z1 = math.exp(lmbda * T / 2)
        z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
        v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(
            2 * kappa / math.pow(sigma_tilde, 2) * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
        return v

    # integrand for the Laplace transform method (cf. (4.9))
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # integration to obtain the option price (cf. (4.9))
    V0 = integrate.quad(integrand, 0, 50)
    return V0


# test parameters
S0 = range(50, 151, 1)
r = 0.05
nu0 = math.pow(0.3, 2)
kappa = math.pow(0.3, 2)
lmdba = 2.5
sigma_tilde = 0.2
T = 1
K = 100
p = 1
R = p + 0.1

V0_Heston = np.empty(101, dtype=float)
for i in range(0, len(S0)):
    V0_Heston[i] = Heston_PCall_Laplace(S0[i], r, nu0, kappa, lmdba, sigma_tilde, T, K, R, p)[0]

plt.plot(S0, V0_Heston, 'red', label='Price of the power call for p = ' + str(p))
plt.legend()
plt.xlabel('Initial stock price S0')
plt.ylabel('Initial option price V0')
plt.show()
