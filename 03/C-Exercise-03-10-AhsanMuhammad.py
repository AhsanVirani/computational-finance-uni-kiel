import numpy as np

def BS_EuOption_MC_CV(S0, r, sigma, T, K, M):
    ## Taken from exercise 02-07-AhsanMuhammad from function: Eu_Option_BS_MC
    # Generate M realizations of X ~ N(0,1)
    X = np.random.normal(loc=0, scale=1, size=M)
    # Generate S_T using the formula given - in sheet 02
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X)

    # Calculate the payoffs for the self-quanto call and the European call - M dimensional
    payoff_self_quanto_call = np.maximum(ST - K, 0) * ST  # Payoff for self-quanto call (S(T) - K) * S(T) - given in question
    payoff_call = np.maximum(ST - K, 0)      # Payoff for European call max(S(T) - K, 0)    

    # Calculate mean payoffs - standard for monte carlo simulation.
    mean_self_quanto = np.mean(payoff_self_quanto_call)
    mean_call = np.mean(payoff_call)

    # Estimate covariance and variance 
    ## by def of covariance E((Y - E(Y))(X - E(X))) - in hints called to use np.cov directly
    covariance_matrix = np.cov(payoff_self_quanto_call, payoff_call)
    covariance_qc_ec = covariance_matrix[0, 1] # off-diagnal entry for 1st col
    variance_ec = covariance_matrix[1, 1]      # diagnal entry     for 2nd col

    # Optimal control variate coefficient
    control_cov_coefficient = covariance_qc_ec / variance_ec

    # discount it to time = 0. Lecture notes, page 11, under: Numerical integration using Monte Carlo - Also done in Exercise_02_07
    call_option_price = np.exp(-r * T) * mean_call

    # Control variate estimator - formulae given in lecture notes as V_hat_cv = (1/N) * summation(f(Xn) - Yn) + E(Y)
    # we discount back the payoffs to present value as with BS model for pricing options and adjust formulae above by control variate c
        # 1. mean_self_quanto - original payoff f(Xn)
        # 2. Yn is the control variate adjusted by control variate coefficient - payoff of standard European option
        # 3. formulae after CV = (np.exp(-r*T) (f(Xn) - c(Yn - E(Yn))) - source https://quant.stackexchange.com/questions/68036/control-variates-option-pricing
    V0_hat_CV = np.exp(-r * T) * (mean_self_quanto - control_cov_coefficient * (mean_call - call_option_price))
    # vanilla self quanto call option initial price without variance control
    V0_BS_MC = np.mean(payoff_self_quanto_call) * np.exp(-r * T)
    
    return V0_hat_CV, V0_BS_MC


# Test the function
S0    = 100
r     = 0.05
sigma = 0.3
T     = 1
K     = 110
M     = 100000
f     = lambda S_t, K: max((S_t - K), 0)

# Compute the price using variance reduction via control variates
V0_CV, V0_BS = BS_EuOption_MC_CV(S0, r, sigma, T, K, M)
print(f"Estimated initial price of the European self-quanto call via CV: {V0_CV} vs initial price of European self-quanto call via BS plain: {V0_BS}")