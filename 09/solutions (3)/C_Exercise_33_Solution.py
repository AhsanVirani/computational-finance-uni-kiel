# C-Exercise 33, SS 2024
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cmath

def Heston_EuCall_MC_Richardson(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m):
    # m is the parameter on the coarse grid
    # increase grid size by a factor of 2
    # fine grid is Euler with grid size * 2, coarse grid is classic Euler from Ex 30
    Delta_t = T/(2*m) 
    # note that we only sample W1 and W2 once and use it for both fine and coarse grid so they are dependent and covariance can reduce variance by 4 * cov(euler on fine grid, euler on coarse grid)
    Delta_W1 = np.random.normal(0,math.sqrt(Delta_t), (M, 2*m))
    Delta_W2 = np.random.normal(0, math.sqrt(Delta_t), (M, 2*m))

    #Initialize matrix which contains the process values
    S_fine = np.zeros((M, 2*m+1))
    gamma_fine = np.zeros((M, 2*m+1))
    S_coarse = np.zeros((M, m + 1))
    gamma_coarse = np.zeros((M, m + 1))

    #Assign first column starting values
    # doesnot matter here the 2nd dimension. For fine grid its 2m and for coarse grid only m
    S_fine[:, 0] = S0 * np.ones(M)
    gamma_fine[:, 0] = gamma0 * np.ones(M)
    S_coarse[:, 0] = S0 * np.ones(M)
    gamma_coarse[:, 0] = gamma0 * np.ones(M)

    # fine grid - apply Euler
    for i in range(0, 2*m):
        gamma_fine[:, i+1] = np.maximum(gamma_fine[:, i] + (kappa - lmbda * gamma_fine[:, i]) * Delta_t + sigma_tilde * np.sqrt(gamma_fine[:, i]) * Delta_W1[:, i],0)
        S_fine[:,i+1] = S_fine[:,i] + r * S_fine[:, i] * Delta_t + S_fine[:, i] * np.sqrt(np.maximum(gamma_fine[:, i], 0)) * Delta_W2[:, i]

    # coarse grid - here we only have m in second dimension
    for i in range(0, m):
        gamma_coarse[:, i+1] = np.maximum(gamma_coarse[:, i] + (kappa - lmbda * gamma_coarse[:, i]) * 2 * Delta_t + sigma_tilde * np.sqrt(gamma_coarse[:, i]) * (Delta_W1[:, 2*i]+Delta_W1[:,2*i+1]),0)
        S_coarse[:,i+1] = S_coarse[:,i] + r * S_coarse[:, i] * 2 * Delta_t + S_coarse[:, i] * np.sqrt(np.maximum(gamma_coarse[:, i], 0)) * (Delta_W2[:, 2*i]+Delta_W2[:, 2*i+1])

    # last column is the payoff so extract them
    payoff_fine = g(S_fine[:,-1])
    payoff_coarse = g(S_coarse[:, -1])
    # richardson is 2 * coarse grid - fine grid
    payoff_richardson = 2 * payoff_fine - payoff_coarse
    # take the mean of the payoff and discount back to t = 0
    Richardson_estimator = math.exp(-r * T) * payoff_richardson.mean()
    epsilon_rich = math.exp(-r * T) * (1.96 * math.sqrt(np.var(payoff_richardson, ddof=1) / M))
    #comparison to the fine estimator
    fine_estimator = math.exp(-r * T) * payoff_fine.mean()
    epsilon_fine = math.exp(-r * T) * (1.96 * math.sqrt(np.var(payoff_fine, ddof=1) / M))
    return Richardson_estimator, epsilon_rich, fine_estimator, epsilon_fine

#to measure how good the MC simulation is, we compute the true value with integral transforms, see lecture notes chapter 7
def Heston_EuCall_Laplace(S0, r, nu0, kappa, lmbda, sigma_tilde, T, K, R):
    # Laplace transform of the function f(x) = (e^(xp) - K)^+ (cf. (7.6))
    def f_tilde(z):
        if np.real(z) > 0:
            return np.power(K, 1 - z) / (z * (z - 1))
        else:
            print('Error')

    # Characteristic function of log(S(T)) in the Heston model (cf. (7.8))
    def chi(u):
        d = cmath.sqrt(
            math.pow(lmbda, 2) + math.pow(sigma_tilde, 2) * (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))))
        n = cmath.cosh(d * T / 2) + lmbda * cmath.sinh(d * T / 2) / d
        z1 = math.exp(lmbda * T / 2)
        z2 = (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * cmath.sinh(d * T / 2) / d
        v = cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T)) * cmath.exp(
            2 * kappa / math.pow(sigma_tilde, 2) * cmath.log(z1 / n)) * cmath.exp(-nu0 * z2 / n)
        return v

    # integrand for the Laplace transform method (cf. (7.9))
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # integration to obtain the option price (cf. (7.9))
    V0 = integrate.quad(integrand, 0, 50)
    return V0[0]

if __name__ == '__main__':
    #Testing Parameters
    S0 = 100
    r = 0.05
    gamma0 = 0.2**2
    kappa = 0.5
    lmbda = 2.5
    sigma_tilde = 1
    T = 1
    M = 10000
    m = 250

    def g(x):
        return np.maximum(x - 100, 0)

    V0_rich, epsilon_rich, V0_fine, epsilon_fine = Heston_EuCall_MC_Richardson(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m)
    Heston_value = Heston_EuCall_Laplace(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, 100, 1.2)
    print("The option price is: " + str(Heston_value))
    print("The Richardson estimate is: " + str(V0_rich))
    print("radius of 95% confidence interval: " + str(epsilon_rich))
    print("The fine grid estimate is: " + str(V0_fine))
    print("radius of 95% confidence interval: " + str(epsilon_fine))
