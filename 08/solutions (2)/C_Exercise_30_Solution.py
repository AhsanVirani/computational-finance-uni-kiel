# C-Exercise 30, SS 2024
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import cmath

def Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m):
    # In Euler method we find gamma, S from i = 1, ..., m i.e. total 250 values each for price and volatility in our case which is an equidistant grid. i.e. i = 1 refers to t1, i = 10 refers to t10 and so on where t1 < t10 < ... < tm
    # M = 10000 means we simulate the equidistant grid of len m 10,000 times.
    
    # Setup parameters as defined in pseudo code for Euler
    Delta_t = T/m
    # M by m i.e. 10000 by 250 variables with ~ N (0, sqrt(deltat)) as given in Euler scheme. Follows from deltaW = Z * sqrt(deltaT)
    # we sample DeltaW1 for gamma stochastic process and DeltaW2 for stock price stochastic process as in Heston model
    Delta_W1 = np.random.normal(0,math.sqrt(Delta_t), (M, m))
    Delta_W2 = np.random.normal(0, math.sqrt(Delta_t), (M, m))

    # Initialize matrix which contains the process values - for both S and gamma
    S = np.zeros((M, m+1))
    gamma = np.zeros((M, m+1))

    # Assign first column starting values. Note that this means for i = 1 where i = 1, ..., m=250. Each i contains 10000 MC simulations
    # hence for the first 10000 values contained in the first dimension of the vector and i = 0 from the 2nd dimension we put init values
    S[:, 0] = S0 * np.ones(M)
    gamma[:, 0] = gamma0 * np.ones(M)

    # Recursively go from i = 1, ..., m. Note that i = 1 already filled as init value above, Hence we now fill from i+1 to m
    for i in range(0, m):
        # Note that in heston model gamma is defined as # dgamma(t) = (kappa - lambda * gamma(t))dt + sqrt(gamma(t)) * sigma_tilde * dW(t)
        # Note that in heston model S is defined as # dS(t) = S(t) * r * dt + S(t) * sqrt(gamma(t)) * dW(t)
        # This means we have a diffusion type of form dX(t) = a(X(t),t)dt + b(X(t),t)dW(t) where:
            # For gamma 
                # a(X(t),t) = (kappa - lambda * gamma(t))
                # b(X(t),t) = sqrt(gamma(t)) * sigma_tilde 
            # For S 
                # a(X(t),t) = S(t) * r
                # b(X(t),t) = S(t) * sqrt(gamma(t))
        # Euler Scheme states that each update equals: Yi = Yi-1 + a(Yi-1,ti-1) * deltaT + a(Yi-1,ti-1) * deltaW
        gamma[:, i+1] = np.maximum(gamma[:, i] + (kappa - lmbda * gamma[:, i]) * Delta_t + sigma_tilde * np.sqrt(gamma[:, i]) * Delta_W1[:, i],0) # gamma is volatility and cant be negatve to use max function to bound it between 0 and positive gamma
        S[:,i+1] = S[:,i] + r * S[:, i] * Delta_t + S[:, i] * np.sqrt(gamma[:, i]) * Delta_W2[:, i]

    # finally, give me the 10000 prices of MC simulations at tm which is time=T i.e. Maturity time
    payoff = g(S[:,-1])
    # take mean of the payoff of 10000 simulations at t = T and discount back to time 0
    MC_estimator = math.exp(-r * T) * payoff.mean()
    
    epsilon = math.exp(-r * T) * (1.96 * math.sqrt(np.var(payoff, ddof=1) / M)) # discount back the epsilon to t = 0
    c1 = MC_estimator - epsilon
    c2 = MC_estimator + epsilon
    return MC_estimator, c1, c2

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

    V0, c1, c2 = Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, g, M, m)
    Heston_value = Heston_EuCall_Laplace(S0, r, gamma0, kappa, lmbda, sigma_tilde, T, 100, 1.2)
    print("The option price is: " + str(Heston_value))
    print("The MC estimate is: " + str(V0))
    print("95% confidence interval: [" + str(c1) + ',' + str(c2) + "].")
