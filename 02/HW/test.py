import numpy as np
import math

def CRR_stock(S_0, r, sigma, T, M):
    """
    Initialize the binomial stock price tree.

    Args:
        S_0 (float): Initial stock price.
        r (float): Annual risk-free interest rate.
        sigma (float): Annual volatility (standard deviation of returns) of the stock.
        T (float): Time to expiration of the option (in years).
        M (int): Number of time steps in the binomial tree.

    Returns:
        S (numpy.ndarray): Matrix of stock prices at each node of the binomial tree.
        u (float): Up factor for stock price.
        d (float): Down factor for stock price.
    """
    delta_t = T / M
    alpha = math.exp(r * delta_t)
    beta = 1 / 2 * (1 / alpha + alpha * math.exp(math.pow(sigma, 2) * delta_t))
    u = beta + math.sqrt(math.pow(beta, 2) - 1)
    d = 1 / u
    # Allocate matrix S to store stock prices
    S = np.zeros((M + 1, M + 1))
    S[0, 0] = S_0

    # Fill the matrix S with stock prices at each node
    for i in range(1, M + 1):
        for j in range(i + 1):
            S[j, i] = S_0 * (u ** (i - j)) * (d ** j)

    return S, u, d

def CRR_AmPut(S_0, r, sigma, T, M, K):
    """
    Calculate the value of an American put option using the Cox-Ross-Rubinstein (CRR) model.

    Args:
        S_0 (float): Initial stock price.
        r (float): Annual risk-free interest rate.
        sigma (float): Annual volatility (standard deviation of returns) of the stock.
        T (float): Time to expiration of the option (in years).
        M (int): Number of time steps in the binomial tree.
        K (float): Strike price of the put option.

    Returns:
        float: Value of the American put option at time 0.
    """
    delta_t = T / M
    S, u, d = CRR_stock(S_0, r, sigma, T, M)
    
    # Calculate risk-neutral probability q
    q = (math.exp(r * delta_t) - d) / (u - d)

    # Initialize the matrix to store option values
    V = np.zeros((M + 1, M + 1))

    # Calculate terminal payoffs at maturity (put option payoff)
    V[:, M] = np.maximum(K - S[:, M], 0)
    print(V)
    # Recursive valuation using backward induction
    for k in range(M - 1, -1, -1):
        for j in range(k + 1):
            # Calculate option value at each node
            payoff = max(K - S[j, k], 0)
            exercise_value = math.exp(-r * delta_t) * (q * V[j, k + 1] + (1 - q) * V[j + 1, k + 1])
            V[j, k] = max(payoff, exercise_value)  # Choose maximum between exercise and hold
    print(V)
    # Return the value of the American put option at time 0
    return V[0, 0]

# Example usage:
S0 = 1.0
r = 0.05
sigma = math.sqrt(0.3)
T = 3.0
M = 3
K = 1.2

# Calculate the value of the American put option
option_value = CRR_AmPut(S0, r, sigma, T, M, K)
print("Value of the American put option:", option_value)
