# C-Exercise 29, SS 2024

import numpy as np
import scipy.stats
import math
import scipy.misc

def EuOptionHedge_BS_MC_IP (St, r, sigma, g, T, t, N):
    h = 0.00000001

    ### calculate derivative with 'scipy.misc.derivative' of the function g(x) at the point S(T). Note that S(T) changes for each simulation.
    def derivative(x):
        return scipy.misc.derivative(g,St * np.exp((r-sigma**2/2)*(T-t)+sigma*np.sqrt(T-t)*x))

    ### alternative way to calculate the derivative with finite differences
    def derivative_alt(x):
        ST = St * np.exp((r-sigma**2/2)*(T-t)+sigma*np.sqrt(T-t)*x)
        return (g((1+h/2)*ST)-g((1-h/2)*ST))/ ( h*ST)


    ### Simulate normal random variables for the simulation of S(T)
    X = np.random.normal(0,1,N)

    ### allocate empty memory space
    delta = np.zeros(N)
    delta2 = np.zeros(N)
    # for LRM
    delta3 = np.zeros(N)

    for i in range(0,N):

        ### infinitesimal perturbation approach
        delta[i] = np.exp(-(np.power(sigma,2)/2)*(T-t)+sigma*np.sqrt(T-t)*X[i]) * derivative(X[i])
        
        ### Bonus(not part of the exercise): Calculate the hedge with finite differences
        ### Note the difference in how the derivative is calculated - given on page 61
        ST_plus = (St+h/2) * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i])
        ST_minus = (St - h / 2) * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i])
        delta2[i] = np.exp(-r*(T-t)) * (g(ST_plus)-g(ST_minus))/ ( h)


        # pricing via likelihood ratio method
        # z_hat(v) = 1/N sum (f(Xn) * dlog(gv(Xn)/dv))
        # we generate stock price using the same X ~ N (0,1)
        S_t     = St * np.exp((r - sigma ** 2 / 2) * (T - t) + sigma * np.sqrt(T - t) * X[i])
        # to calculate z'(v) we samply Y ~ (log(s)/sqrt(sigma ** 2 * T),1) distribution
        mean_sampling_dist = np.log(S_t) / (sigma * np.sqrt(T))
        # Generate N samples from the normal distribution with the calculated mean and unit variance
        Y = np.random.normal(mean_sampling_dist, 1, N)

        # calculate f_x and gx_diff from Y
        f_x     = np.exp(-r * T) * g(np.exp((r - sigma ** 2 /2) * T + sigma * np.sqrt(T) * Y[i]))
        gx_diff = (Y[i] / (S_t * np.sqrt((sigma ** 2) * T))) - (np.log(S_t) / (S_t * (sigma ** 2) * T))

        delta3[i] = f_x * gx_diff
    return (np.mean(delta),np.mean(delta2), np.mean(delta3))

St = 100
r = 0.05
sigma = 0.2
T = 1
N = 10000
t = 0

def g(x):
    return np.maximum(x-90,0)

(pertub, findiff, lrm) = EuOptionHedge_BS_MC_IP (St, r, sigma, g, T, t, N)

print('Hedge with infinitesimal perturbation:'+str(pertub)+ '\nHedge with finite difference approach:'+str(findiff) + '\nHedge with likelihood ratio method:'+str(lrm))


#Hedge according to p.48 in lecture notes
def EuCallHedge_BlackScholes(t, S_t, r, sigma, T, K):
    d_1 = (math.log(S_t / K) + (r + 1 / 2 * math.pow(sigma, 2)) * (T - t)) / (sigma * math.sqrt(T - t))
    return scipy.stats.norm.cdf(d_1)

print('Hedge using the BS-Formula:'  + str(EuCallHedge_BlackScholes(t, St, r, sigma, T, 100)))

