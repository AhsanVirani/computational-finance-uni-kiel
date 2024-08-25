import numpy as np
def Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lamb, sigma_tilde, T, g, M, m):
    dt = T / m  # Time step size
    S_paths = np.zeros((M, m+1))  # Matrix to store stock price paths
    V_paths = np.zeros((M, m+1))  # Matrix to store variance paths
    
    # Initial conditions
    S_paths[:, 0] = S0
    V_paths[:, 0] = gamma0
    
    # Generate independent random numbers for the two Wiener processes
    Z1 = np.random.standard_normal((M, m))
    Z2 = np.random.standard_normal((M, m))
    
    for i in range(1, m + 1):
        S_prev = S_paths[:, i-1]  # Stock price at the previous time step
        V_prev = V_paths[:, i-1]  # Variance at the previous time step
        
        # Euler discretization for variance path
        V_paths[:, i] = V_prev + kappa * (lamb - V_prev) * dt + sigma_tilde * np.sqrt(V_prev) * np.sqrt(dt) * Z2[:, i-1]
        
        # Ensure that variance remains positive
        V_paths[:, i] = np.maximum(V_paths[:, i], 0)
        
        # Euler discretization for stock price path
        S_paths[:, i] = S_prev + S_prev * r * dt + S_prev * np.sqrt(V_prev) * np.sqrt(dt) * Z1[:, i-1]
    
    # Compute the payoff at maturity
    payoff = g(S_paths[:, -1])
    
    # Discount the payoff to present value
    discounted_payoff = np.exp(-r * T) * payoff
    
    # Compute the price of the option
    V0 = np.mean(discounted_payoff)
    
    # Compute the standard error and the 95% confidence interval
    std_err = np.std(discounted_payoff) / np.sqrt(M)
    ci = 1.96 * std_err  # 1.96 is the z-score for a 95% confidence interval
    
    return V0, V0 - ci, V0 + ci

# Define the parameters for the test
S0 = 100  # Initial stock price
r = 0.05  # Risk-free interest rate
gamma0 = 0.2 ** 2  # Initial variance
kappa = 2.5  # Rate of mean reversion of variance
lamb = 0.5  # Long-term variance
sigma_tilde = 1  # Volatility of volatility
T = 1  # Time to maturity
g = lambda x: np.maximum(x - 100, 0)  # Payoff function for a European call option
M = 10000  # Number of Monte Carlo samples
m = 250  # Number of time steps for Euler method

# Compute the option price and confidence interval
V0, c1, c2 = Heston_EuCall_MC_Euler(S0, r, gamma0, kappa, lamb, sigma_tilde, T, g, M, m)

# Print the results
print("Estimated option price V0:", V0)
print("95% confidence interval:", (c1, c2))