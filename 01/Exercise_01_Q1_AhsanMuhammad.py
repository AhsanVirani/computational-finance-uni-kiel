# Authored By: Ahsan Muhammad (1183091)
# Dated: 02.05.2024
# Group 12

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#######################################################
# A - CRR Stock Price Binomial Tree Init
# Referenced from 
#######################################################
def CRR_stock(S_0, r, sigma, T, M):
    """
    Generate a Cox-Ross-Rubinstein (CRR) binomial tree of stock prices.

    Args:
        S_0 (float): Initial stock price (at time 0)
        r (float): Risk-free interest rate per annum
        sigma (float): Volatility of the stock price per annum
        T (float): Time to maturity of the option (in years)
        M (int): Number of time steps in the binomial tree

    Returns:
        numpy.ndarray: Stock price matrix representing the CRR binomial tree
    """
    timestep = T / M
    beta = (1/2) * (math.exp(-r * timestep) + math.exp((r + sigma ** 2) * timestep))
    u = beta + math.sqrt(beta ** 2 - 1)
    d = 1 / u
    
    # Initialize stock price matrix
    dim = M + 1
    S = np.zeros((dim, dim))
    
    # Fill the stock price matrix using the CRR model
    for i in range(dim):
        for j in range(i + 1):
            S[i, j] = S_0 * (u ** j) * (d ** (i - j))
    
    # Return the CRR stock price tree
    return S

#######################################################
# B - CRR European Call Option Valuation
#######################################################
def CRR_EuCall(S_0, r, sigma, T, M, K):
    """
    Calculate the approximate price of a European call option using the Cox-Ross-Rubinstein (CRR) binomial model.

    Args:
        S_0 (float): Initial stock price (at time 0)
        r (float): Risk-free interest rate per annum
        sigma (float): Volatility of the stock price per annum
        T (float): Time to maturity of the option (in years)
        M (int): Number of time steps in the binomial model
        K (float): Strike price of the option

    Returns:
        float: Approximate price of the European call option using the CRR model
    """
    timestep = T / M
    u = np.exp(sigma * np.sqrt(timestep))
    d = 1 / u
    q = (np.exp(r * timestep) - d) / (u - d)
    
    # Generate the stock price tree using CRR model
    stock_prices = CRR_stock(S_0, r, sigma, T, M)
    
    # Initialize option values at maturity (payoff) - For call option max(S_t - K, 0) at leaf nodes
    option_values = np.maximum(stock_prices[M, :] - K, 0)
    
    # Backward recursion to calculate option price
    for i in range(M - 1, -1, -1):
        for j in range(i + 1):
            option_values[j] = np.exp(-r * timestep) * (q * option_values[j] + (1 - q) * option_values[j + 1])
    
    # Return the option price at time 0 - i.e., the root node
    return option_values[0]

#######################################################
# C - BlackScholes
#######################################################
def BlackScholes_EuCall(t, S_t, r, sigma, T, K):
    """
    Calculate the fair price of a European Call option using the Black-Scholes formula.

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
    # parameter calculation from h.w sheet
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C_t = S_t * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return C_t


#######################################################
# D - Compare CRR & BlackScholes & Plot Error
#######################################################
# Parameters for the comparison
S_0 = 100   # Initial stock price
r = 0.03    # Annual risk-free interest rate
sigma = 0.3 # Annual volatility
T = 1       # Time to maturity (in years)
M = 100     # Number of time steps in the CRR model
K_range = range(70, 201, 1)  # Range of strike prices K to compare - In steps of 1 from 70 to 200

# Calculate CRR model prices and corresponding BS model prices
crr_prices = [CRR_EuCall(S_0, r, sigma, T, M, K) for K in K_range]
bs_prices = [BlackScholes_EuCall(0, S_0, r, sigma, T, K) for K in K_range]

# Calculate errors between CRR model and BS model prices
errors = np.array(crr_prices) - np.array(bs_prices)

# Plot the pricing errors
plt.figure(figsize=(12, 6))
plt.plot(K_range, errors, label='Error', color='b', marker='o')
plt.xlabel('Strike Price (K)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Pricing Error: CRR Model vs. Black-Scholes Model', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
