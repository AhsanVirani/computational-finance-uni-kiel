## Some useful points
# perpetual option never expires hence is time-independent.
# the optimal exercise threshold (also called the exercise boundary) can be derived from model parameters to maximize the value of the option
    # optimal exercise threshold derived by value matching condition S = S^* - value of the option when exercised must equal to the value of the option if held. Indifferent to holding or exercising
    # For S >= S* the intrinsic value of Am.Perp.Put given by (K-S(T))^+ <= value from pde approach lets say (exercise value) - better to hold
    # For S <= S* the intrinsic value of Am.Perp.Put given by (K-S(T))^+ > value from pde approach lets say (exercise value)  - better to exercise

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def BS_AmPerpPut_ODE(S_max, N, r, sigma, K):
    g = lambda S: np.maximum(K - S, 0) # the intrinsic value of Am.Perp.Put option

    S_grid = np.linspace(0, S_max, N + 1) # creating an equidistant grid of the stock price from 0 to S_max
    v_grid = np.zeros_like(S_grid)        # v_grid to store Am.Perp.Put option init value at each stock price corresponding to index value of S_grid

    # Define a 2-dimensional system of 1st-order ODEs corresponding to the given 2nd-order ODE
    fun = lambda x, v: np.array([v[1], 2 * r / (sigma ** 2 * x ** 2) * (v[0] - x * v[1])])
    x_star = 2 * K * r / (2 * r + sigma ** 2) # optimal exercise threshold - given in question

    # For x <= x_star, the option value v(x) is given by the payoff
    # apply only to v_grid where X <= x*. We exercise immediately as the intrinsic value (K - ST)^+ > V(T) given by solving BlackScholes pde
    v_grid[S_grid <= x_star] = g(S_grid[S_grid <= x_star])

    # For x > x_star, we integrate from x_star to S_max defined in t_span. This is the first argument x in the fun lambda func
        # we set initial conditions i.e. defined as v in fun lambda func. v[0] is value of option i.e. g(x_star) and v[1] is dv/dx* = -1 defined in y0
    # defining t_eval of the solutions are only stored at points where X > X^*
    result = solve_ivp(fun=fun, t_span=(x_star, S_max), y0=[g(x_star), -1], t_eval=S_grid[S_grid > x_star])
    # we need to apply the results of the initial value given by solving the ODE to indexes in V_grid where the price X > X^*. Filtering at S_grid > x_star does the trick i.e. index in S_grid corresponds to the specific value of option at that particular index in V_grid
    v_grid[S_grid > x_star] = result.y[0]

    return S_grid, v_grid



S_max = 200
N = 200
r = 0.05
sigma = np.sqrt(0.4)
K = 100

S_grid, v_grid = BS_AmPerpPut_ODE(S_max, N, r, sigma, K)

plt.figure(figsize=(8, 6))
plt.plot(S_grid, v_grid)
plt.axvline(2 * K * r / (2 * r + sigma ** 2), linestyle='--', color='red', alpha=0.5)
plt.grid(alpha=0.4)
plt.legend(['Perpetual American Put', 'Optimal Exercise Barrier'])
plt.xlabel('S')
plt.ylabel('V')
plt.show()
