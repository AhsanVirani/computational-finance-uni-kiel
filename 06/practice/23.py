# C-Exercise 23, SS 2024

# importing libraries
import math
import cmath
import scipy.integrate as integrate
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt



# computes the price of a european call/put option in the Black-Scholes model via the Laplace transform approach
def BS_Eu_Laplace(S0, r, sigma, T, K, R):
    """
    Price European call and put options using the Black-Scholes model
    and Laplace transform methods.

    Parameters:
    -----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    K : float or array-like
        Strike price(s) of the option(s).
    R : float
        Damping factor used in the Laplace transform.
        
    Returns:
    --------
    option_prices : float or array-like
        Price(s) of the option(s) for the given strike price(s).
    """
    # Laplace transform of the function of EuCall payoff f(x) = (e^x - K)^+
    # Also holds for EuPut option with function f(x) = (e^x - K)^+
    # This is the only function that would require changing depending on the type of payoff
    # (eq.4.6)
    # * CHANGE HERE * - payoff dependent
    def f_tilde(z):
        return (K ** (1 - z)) / (z * (z - 1))
    
    # characteristic function of log(S(T)) in the Black-Scholes model. Putting t = 0 yields the below eq from (4.7)
    # * DO NOT CHANGE * - payoff independent 
    def chi(u):
        return cmath.exp(
            complex(0, 1) * u * (math.log(S0) + r * T) - (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * math.pow(
                sigma, 2) / 2 * T)
        
    # Integrand for the Laplace transform method (eq.4.4)
    # We integrate w.r.t u hence u will be tuned during integration
    # z = R + iu for f_tilde and z = u - iR for the characterstic func
    # * DO NOT CHANGE *
    def integrand(u):
        return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

    # option price - we can integrate from 0 to inf or place a higher upper limit - lets do to infinity for now
    V0 = integrate.quad(integrand, 0, np.inf)
    return V0

# computes the price of a call/Put in the Black-Scholes model - for comparison purposes against our laplace transform option price
# eq 3.23
# * DO NOT CHANGE *
def Eu_BlackScholes(t, S_t, r, sigma, T, K, type=1):
    """
    Calculate the price of a European call or put option using the Black-Scholes formula.

    Parameters:
    -----------
    t : float
        Current time (usually 0).
    S_t : float
        Current stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    K : float
        Strike price of the option.
    type : int, optional (default=1)
        Option type: 1 for call option, 0 for put option.

    Returns:
    --------
    C : float
        Price of the European call or put option.
    phi : float
        Cumulative distribution function (CDF) value for d_1 (for calls) or -d_2 (for puts).

    Notes:
    ------
    This function uses the Black-Scholes formula to calculate the price of a European option.
    The formula is based on the assumption that the option follows a lognormal distribution
    and that markets are efficient.
    """
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    d_2 = d_1 - sigma * math.sqrt(T - t)
    
    if type:
        phi = scipy.stats.norm.cdf(d_1)
        C = S_t * phi - K * math.exp(-r * (T - t)) * scipy.stats.norm.cdf(d_2)
    else:
        phi = scipy.stats.norm.cdf(-d_2)
        C = K * math.exp(-r * (T - t)) * phi - S_t * scipy.stats.norm.cdf(-d_1)
    return C, phi


# test parameters
S0 = range(50, 151, 1)
r = 0.03
sigma = 0.2
T = 1
K = 110


# for EuCall R > 1 for EuPut R < 0. Determine by solving for f_tilde the value of R = Re(z) for any other payoff structure 
# * CHANGE HERE *
R_call = 1.1 # for call R > 1
R_put  = -0.1 # for put R < 0

# price of EU Call option by passing R_call = 1.1 to BS_Eu_Laplace func and setting type = 1 for Eu_BlackScholes (call) for comparison
print(
    'Price of European call by use of Laplace transform approach: ' + str(BS_Eu_Laplace(S0[50], r, sigma, T, K, R_call)[0]))
print('Price of European call by use of the BS-formula: ' + str(Eu_BlackScholes(0, S0[50], r, sigma, T, K, 1)[0]))

# price of EU Put option by passing R_put = -0.1 to BS_Eu_Laplace func and setting type = 0 for Eu_BlackScholes (put) for comparison
print(
    'Price of European put by use of Laplace transform approach: ' + str(BS_Eu_Laplace(S0[50], r, sigma, T, K, R_put)[0]))
print('Price of European put by use of the BS-formula: ' + str(Eu_BlackScholes(0, S0[50], r, sigma, T, K, 0)[0]))


V0_call = np.empty(101, dtype=float)
V0_put = np.empty(101, dtype=float)
for i in range(0, len(S0)):
    V0_call[i] = BS_Eu_Laplace(S0[i], r, sigma, T, K, R_call)[0] # tree of init option price of EUCall by Laplace
    V0_put[i] = BS_Eu_Laplace(S0[i], r, sigma, T, K, R_put)[0]   # tree of init option price of EUPut by Laplace

plt.plot(S0, V0_call, 'green', label='Price of the european call')
plt.plot(S0, V0_put, 'red', label='Price of the european put')
plt.legend()
plt.xlabel('Initial stock price S0')
plt.ylabel('Initial option price V0')
plt.show()