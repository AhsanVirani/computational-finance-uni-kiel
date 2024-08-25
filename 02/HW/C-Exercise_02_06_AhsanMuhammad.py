# Importing libraries
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt

import numpy as np
import math


def CRR_stock(S_0, r, sigma, T, M):
    """
    Function to initialise CRR_stock tree. 
    This code is is reference from C-Exercise_01_Solution.py

    Args:
        S_0 (float): Initial stock price
        r (float): risk free interest rate
        sigma (float): volatility of the stock
        T (int): Time to maturity of the option
        M (int): Number of sub branches in the tree

    Returns:
        _type_: _description_
    """
    # compute values of u, d and q
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    # allocate matrix S
    S = np.empty((M + 1, M + 1))

    # fill matrix S with stock prices
    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)
    
    # Return CRR binomial tree S, up-factor u and down-factor d
    return S, u, d

def CRR_AmEuPut(S_0, r, sigma, T, M, K, EU):
    """
    CRR AmEuPut function calculates the price of European put option if EU=1 and American put option if EU=0
    Some parts of this function has been referenced from CRR_EuCall from solution exercise from C-Exercise_01_Solution.py
    Args:
        S_0 (float): Initial stock price
        r (float): risk free interest rate
        sigma (float): volatility of the stock
        T (int): Time to maturity of the option
        M (int): Number of sub branches in the tree
        K (float): option strike price
        EU (_type_): calculates European put option price if EU = 1 and American put option if EU=0

    Raises:
        ValueError: EU can either be 0 or 1. Raises ValueError if EU doesn't abide by the rules

    Returns:
        V[0]: Value of the put option
    """
    delta_t = T / M
    S, u, d = CRR_stock(S_0, r, sigma, T, M)
    
    # calculating q (the risk neutral probability) - lectures pg. 5
    q = (math.exp(r * delta_t) - d) / (u - d)

    # V will contain the call prices
    V = np.zeros((M + 1, M + 1))
    
    # compute the value of the call at time T - defined as max(K - S_T, 0)
    V[:, M] = np.maximum(K - S[:, M], 0)

    # define recursion function
    def g(k):
        return math.exp(-r * delta_t) * (q * V[1:k + 1, k] +
                                         (1 - q) * V[0:k, k])
    # compute call prices at t_i
    for k in range(M, 0, -1):
        # American Put Option Case 1:
        if EU == 0:
            # payoff of put option at the preceding nodes - max(K - S_T, 0) - call it payoff_1
            payoff_1 = np.maximum(K - S[0:k, k - 1], 0)
            # payoff calculated by func g(k) - call it payoff_2
            payoff_2 = g(k)
            # payoff of the nodes is max of payoff_1 and payoff_2. Selecting payoff_1 means we exercise the option
            payoff_fin = np.maximum(payoff_1, payoff_2)
        # European Put Option Case 2:
        elif EU == 1:
            payoff_fin = g(k)
        # Error Case 3:
        else:
            raise ValueError("EU parameter can either be 1 (for European option) or 0 (for American option)")
        V[0:k, k - 1] = payoff_fin
    
    # return the price of the call at time t_0 = 0
    return V[0, 0]
    
def BlackScholes_EuPut(t, S_t, r, sigma, T, K):
    """
    Calculate the fair price of a European Call option using the Black-Scholes formula.
    This code is is reference from C-Exercise_01_Solution.py function BlackScholes_EuCall
    Args:
        t (float): Current time (0 ≤ t ≤ T)
        S_t (float): Stock price at time t
        r (float): Risk-free interest rate (annual rate, continuously compounded)
        sigma (float): Volatility of the stock price (annualized)
        T (float): Time to maturity of the option (T - t)
        K (float): Strike price of the option

    Returns:
        float: Fair price of the European Call option
    """
    d1 = (math.log(S_t / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * math.sqrt(T - t))
    d2 = d1 - sigma * math.sqrt(T - t)
    phi_d1 = scipy.stats.norm.cdf(d1)  # CDF of d1
    phi_d2 = scipy.stats.norm.cdf(d2)  # CDF of d2
    # Changing this equation for put option as given in the h.w sheet
    P = K * math.exp(-r * (T - t)) * (1 - phi_d2) - S_t * (1 - phi_d1)
    return P


# # test parameters
# S_0 = 100
# r = 0.05
# sigma = 0.3
# T = 1
# K = range(10, 501, 1)
# M = 120

# V_0_EU = np.empty(491, dtype=float)
# V_0_AM = np.empty(491, dtype=float)
# V_0_BS = np.empty(491, dtype=float)

# for i in range(0, len(K)):
#     V_0_EU[i] = CRR_AmEuPut(S_0, r, sigma, T, M, K[i], EU=1)
#     V_0_AM[i] = CRR_AmEuPut(S_0, r, sigma, T, M, K[i], EU=0)
#     V_0_BS[i] = BlackScholes_EuPut(0, S_0, r, sigma, T, K[i])


# # Part A - Plotting EU put option pricing
# plt.plot(V_0_EU, 'r-', label='European Put Option Pricing')
# # Part B - Plotting BS put option pricing on the same shart
# plt.plot(V_0_BS, 'b--', label='Black-Scholes Put Option Pricing')
# plt.xlabel('Number of Steps M')
# plt.ylabel('Price of the option')
# plt.legend()
# plt.show()

# # Part C - priting to console the American Option Pricing 
# print(V_0_AM)

# V_0 = CRR_AmEuPut(S_0, r, sigma, T, 500, 100, EU = 0)
# print('The price of the American put option for the test parameters is given by: ' + str(V_0))

# part c)
# test parameters
S_0 = 100
r = 0.05
sigma = 0.3
T = 1
M = range(10, 501, 1)
K = 120

EuPutprices = np.zeros(501)
for i in M:
    EuPutprices[i] = CRR_AmEuPut(S_0, r, sigma, T, i, K, EU = 1)
BSprice = BlackScholes_EuPut(0, S_0, r, sigma, T, K)

plt.plot(np.arange(10,501),EuPutprices[10:], 'r', label = 'Binomial model price')
plt.plot(np.arange(10,501), BSprice * np.ones(491),'b', label = 'Black-Scholes price')
plt.xlabel('number of steps')
plt.ylabel('price')
plt.legend()
plt.show()

V_0 = CRR_AmEuPut(S_0, r, sigma, T, 500, K, EU = 0)
print('The price of the American put option for the test parameters is given by: ' + str(V_0))

