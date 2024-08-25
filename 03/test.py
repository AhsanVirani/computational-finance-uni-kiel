import numpy as np

def BS_EuOption_MC_CV(S0, r, sigma, T, K, M):
    np.random.seed(42)  # For reproducibility
    
    # Step 1: Simulate paths for S(T)
    Z = np.random.standard_normal(M)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Step 2: Compute the payoff of the self-quanto call option
    payoff_self_quanto = (ST - K) * ST
    
    # Step 3: Compute the payoff of the European call option
    payoff_call = np.maximum(ST - K, 0)
    
    # Step 4: Estimate the covariance and variance needed for the control variate
    cov_matrix = np.cov(payoff_self_quanto, payoff_call)
    cov = cov_matrix[0, 1]
    var_control = cov_matrix[1, 1]
    beta = cov / var_control
    
    # Step 5: Apply the control variate technique
    call_price = np.mean(payoff_call) * np.exp(-r * T)
    control_variate_estimator = np.mean(payoff_self_quanto) - beta * (np.mean(payoff_call) - call_price)
    V0 = control_variate_estimator * np.exp(-r * T)
    
    # Plain Monte Carlo simulation for comparison
    plain_MC_estimator = np.mean(payoff_self_quanto) * np.exp(-r * T)
    
    return V0, plain_MC_estimator

# Parameters
S0 = 100
r = 0.05
sigma = 0.3
T = 1
K = 110
M = 100000

# Compute the option price using Monte Carlo with control variates
V0, plain_MC_estimator = BS_EuOption_MC_CV(S0, r, sigma, T, K, M)

print(f"Option price with control variates: {V0:.4f}")
print(f"Option price with plain Monte Carlo: {plain_MC_estimator:.4f}")
