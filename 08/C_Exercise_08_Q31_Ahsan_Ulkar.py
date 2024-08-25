######### C-31 ############import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np


def Sim_Paths_GeoBM(X0, mu, sigma, T, N):

    dt = T / N  # Time step size
    t = np.linspace(0, T, N+1)  # Time grid
    
    # Generate Brownian increments
    Delta_W = np.random.normal(0, math.sqrt(dt), N)  # Brownian increments
    W = np.insert(np.cumsum(Delta_W), 0, 0)  # Brownian path (cumulative sum with 0 at the start)

    # Exact solution for Geometric Brownian Motion
    X_exact = X0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    # Euler method for Geometric Brownian Motion
    X_Euler = np.zeros(N+1)
    X_Euler[0] = X0
    for i in range(1, N+1):
        X_Euler[i] = X_Euler[i-1] + mu * X_Euler[i-1] * dt + sigma * X_Euler[i-1] * Delta_W[i-1]

    # Milshtein method for Geometric Brownian Motion
    X_Milshtein = np.zeros(N+1)
    X_Milshtein[0] = X0
    for i in range(1, N+1):
        X_Milshtein[i] = X_Milshtein[i-1] + mu * X_Milshtein[i-1] * dt + sigma * X_Milshtein[i-1] * Delta_W[i-1] \
                         + 0.5 * sigma*2 * X_Milshtein[i-1] * (Delta_W[i-1]*2 - dt)

    return X_exact, X_Euler, X_Milshtein

# Define the parameters for the test
X0 = 100  # Initial value of the process
mu = 0.1  # Drift coefficient
sigma = 0.3  # Diffusion coefficient
T = 1  # Time horizon
Ns = [10, 100, 1000, 10000]  # Different numbers of time steps to test

# Simulate and plot the paths for different values of N
for N in Ns:
    X_exact, X_Euler, X_Milshtein = Sim_Paths_GeoBM(X0, mu, sigma, T, N)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, T, N+1), X_exact, label='Exact Solution')
    plt.plot(np.linspace(0, T, N+1), X_Euler, label='Euler Method', linestyle='--')
    plt.plot(np.linspace(0, T, N+1), X_Milshtein, label='Milshtein Method', linestyle='-.')
    plt.title(f'Simulated Paths of Geometric Brownian Motion (N={N})')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.legend()
    plt.show()