# C-Exercise 31, SS 2024
import numpy as np
import math
import matplotlib.pyplot as plt

def Sim_Paths_GeoBM(X0, mu, sigma, T, N):
    # Notice here we only have N and no m as in previous question, hence we use N by 1 vector where N is the equidistant grid now
    Delta_t = T/N
    Delta_W = np.random.normal(0, math.sqrt(Delta_t), (N,1))
    #Initialize vectors with starting value
    X_exact = X0 * np.ones(N+1)
    X_Euler = X0 * np.ones(N + 1)
    X_Milshtein = X0 * np.ones(N + 1)

    # Recursive simulation according to the algorithms in Section 4.2 using identical Delta_W
    # Our dX(t) = u * X(t) * dt + sigma * X(t) * dW(t)
        # a(X(t), t) = u * X(t)
        # b(X(t), t) = sigma * X(t)
    for i in range(0, N):
        X_exact[i+1] = X_exact[i] * np.exp((mu- math.pow(sigma,2)/2)*Delta_t + sigma * Delta_W[i])
        X_Euler[i+1] = X_Euler[i] * (1 + mu * Delta_t + sigma * Delta_W[i])   # dX(t) = u * X(t) * dt + sigma * X(t) * dW(t) Euler scheme outputs X(ti) = X(ti-1) + (a(X(t),t), b(X(t),t) = X(ti-1) + mu * X(ti-1) * dt + sigma * X(ti-1) * W(t) = X(ti-1) * (1 + mu * dt + sigma * dW)
        X_Milshtein[i+1] = X_Milshtein[i] * (1+mu*Delta_t + sigma*Delta_W[i] + math.pow(sigma,2)/2*(math.pow((Delta_W[i]), 2)- Delta_t)) # X(ti-1) + mu * X(ti-1) * dt + sigma * X(ti-1) * W(t) + sigma * sigma * 0.5 * X(ti-1) * ((dW) ** 2 - dt) # sigma * sigma * 0.5 * X(ti-1) follows from b=sigma * X(t) * b'

    return X_exact, X_Euler, X_Milshtein

#test parameters
X0 = 100
mu = 0.1
sigma = 0.3
T = 1
N = np.array([10,100,1000,10000])

#plot
plt.clf()
for i in range(0,4):
    X_exact, X_Euler, X_Milshtein = Sim_Paths_GeoBM(X0, mu, sigma,T, N[i])
    plt.subplot(2,2,i+1)
    plt.plot(np.arange(0, N[i]+1)*T/N[i], X_exact, label = 'Exact Simulation')
    plt.plot(np.arange(0, N[i]+1) * T / N[i], X_Euler, 'red', label = 'Euler approximation')
    plt.plot(np.arange(0, N[i]+1) * T / N[i], X_Milshtein, 'green', label = 'Milshtein approximation')
    plt.xlabel('t')
    plt.ylabel('X(t)')
    plt.title('N=' + str(N[i]))
    plt.legend()

plt.show()