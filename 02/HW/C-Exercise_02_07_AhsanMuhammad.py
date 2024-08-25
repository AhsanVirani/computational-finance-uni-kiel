# Importing libraries
import numpy as np
import math

def Eu_Option_BS_MC (S0, r, sigma, T, K, M, f):
    """
    The function calculations the European call option price via MC simulation
    and returns the price of the option, together with the 95% CI interval

    Args:
        S_0 (float): Initial stock price
        r (float): risk free interest rate
        sigma (float): volatility of the stock
        T (int): Time to maturity of the option
        M (int): Number of sub branches in the tree
        K (int): strike price of the stock
        f (function pointer): payoff of call option function pointer

    Returns:
        V0, c1, c2: Price of the call option, confidence interval lower and upper bands
    """
    # M realisations of X ~ N(0,1)
    X = np.random.normal(loc=0, scale=1, size=M)
    
    # Generating S_T using the formulae given - Given in the question sheet
    S_T = [S0 * math.exp((r - sigma ** 2 / 2) * T + sigma * math.sqrt(T) * x) for x in X]
    
    # applying function f to each S_T. Lecture 2 eq 2.1 pg 11
    Eq_S_T = np.mean([f(s_t, K) for s_t in S_T])
    
    # discount it to time = 0. Lecture notes, page 11, under: Numerical integration using Monte Carlo
    V0 = math.exp(-r * T) * Eq_S_T
    
    # Confidence Interval 
    ## variance / standard deviation - eq 2.5 pg. 13
    var = np.var([f(s_t, K) for s_t in S_T], ddof=1)
    
    ## Confidence Interval - eq 2.6 pg. 13
    ci_lower_band = Eq_S_T - 1.96 * math.sqrt(var / len(X))
    ci_upper_band = Eq_S_T + 1.96 * math.sqrt(var / len(X))
    
    # return V0, confidence intervals
    return V0, ci_lower_band, ci_upper_band

# Testing code
S0    = 110
r     = 0.04
sigma = 0.2
T     = 1
K     = 100
M     = 10000
# defining lambda function of payogg
f     = lambda S_t, K: max((S_t - K), 0)

# Calling the function and saving return in a variable
Eu_call_price_BS_MC = Eu_Option_BS_MC(S0, r, sigma, T, K, M, f)
# Printing Results
print(f"V(0): {Eu_call_price_BS_MC[0]} with 95% Confidence Interval: {Eu_call_price_BS_MC[1]}, {Eu_call_price_BS_MC[2]}")