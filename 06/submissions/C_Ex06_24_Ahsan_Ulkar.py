# Ahsan Muhammad (1183091)
# Ulkar Jafarova (1190872)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# We need to solve ODE from x_star to Smax because x >= x* boundary condition. But we have a second order ODE so we must convert it to a first order ODE
# Then we use solve_ivp to solve the system of ODE which takes the function, boundary condition, an initial condition and option t_eval
    # Our function is
    # Boundary conditions from x_start to Smax
    # initial condition

def BS_AmPerpPut_ODE(S_max, N, r, sigma, K):
    # Define the grid which is equidistant
    S0 = np.arange(0, S_max + S_max/N, S_max/N)
    # x <= x_star then we value like standard put opton (K - S(T))+ - we initialise here and later add to it the ode solved solution for x >= x_star
    # by eq 2 in notes 
    v = np.maximum(K - S0, 0)
    
    # x_star as defined in the question (2kr/(sr+sigma^2))
    x_star = (2 * K * r) / (2 * r + sigma**2)
 
    # Define the ODE system
    def ode_system(x, y):
        # x is the stock price, y is the (value of option, first derivative)
        # let u = dv/dS, du/dS = d^2v/dS^2, insert this in the second order ODE equation. Solve for du/dS to get (2/sigma ** 2) * (rv - rSu)
        d2v_dS2 = (r * y[0] - r * x * y[1]) * (2 / (sigma**2 * x**2)) # du/dS as shown above
        return (y[1], d2v_dS2)

    # Solve the ODE system using solve_ivp from x_star to S_max - as given in the question - solve ode when x >= x*
    # by eq 1 in notes
    solution = solve_ivp(ode_system, [x_star, S_max], [K - x_star, -1]) 
    # Solution contains t and y - stock and option values respectively. y is in a list of list.
    
    # match the grid points S
    v_interpolated = np.interp(S0, solution.t[::-1] , solution.y[0][::-1])
    
    # Apply the boundary condition S >= x_star - Sol given by ODE
    v[S0 > x_star] = v_interpolated[S0 > x_star]
    return v, S0

# Define parameters
S_max = 200   
N = 200      
r = 0.05      
sigma = np.sqrt(0.4) 
K = 100       

# Calculate the option price and also get the equidistant stock prices
option_prices, stock_prices = BS_AmPerpPut_ODE(S_max, N, r, sigma, K)

# Plot the result
plt.plot(stock_prices, option_prices, label='Perpetual Am_Put Option Valuation')
plt.xlabel('Stock Price S(T)')
plt.ylabel('Option Price V(T)')
plt.title('Perpetual Am_Put Option Valuation in BS Model')
plt.legend()
plt.show()


"""
Explanation:
1. For x <= x* The value of the option is simply the payoff - starts at 100 decreases to 80 for price 0 to 20

2. For x > x*  The value should remain constant as solved by ODE reflecting the intrinsic value adjusted for the optimal stopping time, 
but since it's perpetual, the holder will optimally exercise at x*
and above that point, it remains constant.  
"""