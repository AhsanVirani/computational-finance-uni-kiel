import numpy as np
import math
import scipy.integrate as integrate
import cmath

def Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m):
    dt = T / m
    sqrt_dt = math.sqrt(dt)

    Delta_W1 = np.random.normal(0, sqrt_dt, (M, m))
    Delta_W2 = np.random.normal(0, sqrt_dt, (M, m))

    S = np.zeros((M, m + 1))
    gamma = np.zeros((M, m + 1))

    S[:, 0] = S0
    gamma[:, 0] = gamma0

    for i in range(m):
        gamma[:, i + 1] = np.maximum(gamma[:, i] + (kappa - lmbda * gamma[:, i]) * dt
                                     + sigma_tilde * np.sqrt(np.maximum(gamma[:, i], 0)) * Delta_W1[:, i], 0)
        S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * np.maximum(gamma[:, i], 0)) * dt
                                      + np.sqrt(np.maximum(gamma[:, i], 0)) * Delta_W2[:, i])

    payoff = g(S[:, -1])
    MC_estimator = np.exp(-r * T) * payoff.mean()

    epsilon = np.exp(-r * T) * 1.96 * np.sqrt(np.var(payoff, ddof=1) / M)
    confidence_interval = (MC_estimator - epsilon, MC_estimator + epsilon)

    return MC_estimator, confidence_interval

def Heston_EuCall_Laplace(S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R):
    def f_tilde(z):
        return np.power(K, 1 - z) / (z * (z - 1))

    def chi(u):
        d = cmath.sqrt(lmbda ** 2 + sigma_tilde ** 2 * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
        n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
        z1 = math.exp(lmbda * T / 2)
        z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
        v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(
            2 * kappa / sigma_tilde ** 2 * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
        return v

    def integrand(u):
        return np.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    V0 = integrate.quad(integrand, 0, 50)[0]

    return V0

# Parameters
S0 = 100
r = 0.05
gamma0 = 0.2 ** 2
kappa = 0.5
lmbda = 2.5
sigma_tilde = 1
T = 1
M = 10000
m = 250

# Define payoff function
def g(x):
    return np.maximum(x - 100, 0)

# Compute Monte Carlo estimate
V0, (c1, c2) = Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m)

# Compute true value using Laplace transform
Heston_value = Heston_EuCall_Laplace(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, 100, 1.2)

# Output results
print(f'The option price via Heston Model: {Heston_value}')
print(f'The MC estimate is: {V0}')
print(f'95% confidence interval: [{c1}, {c2}]')
