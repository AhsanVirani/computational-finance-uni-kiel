# C-Exercise 09, SS 2024

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# pdf of truncated exponential distribution
def TruncNormal_pdf(x, a, b, mu, sigma):
    if a <= x and x <= b:
        return norm.pdf(x, mu, sigma) / (norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma))
    else:
        return 0


def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    # compute C as the maximum of f(x)/g(x), such that f(x) <= C*g(x)
    C = TruncNormal_pdf(mu, a, b, mu, sigma) * (b - a)

    # function to generate a single sample of the distribution
    def SingleSample():
        # set run parameter for while-loop
        success = False
        while not success:
            # generate two U([0,1]) random variables
            U = np.random.uniform(size=(2, 1))
            # scale one of them to the correct interval
            Y = U[0] * (b - a) + a
            # check for rejection/acceptance
            # when the sample gets rejected the while loop will generate a new sample and will check again
            success = ((C * U[1]/ (b - a)) <= TruncNormal_pdf(Y, a, b, mu, sigma))
        return Y[0]

    # use function SingleSample N times to generate N samples
    X = np.empty(N, dtype=float)
    for i in range(0, N):
        X[i] = SingleSample()

    return X


# test parameters
a = 0
b = 2
mu = 0.5
sigma = 1
N = 10000

X = Sample_TruncNormal_AR(a, b, mu, sigma, N)
# plot histogram
plt.hist(X, 50, density=True)

# plot exact pdf
x = np.linspace(a-a/20,b+b/20,1000)
pdf_vector = np.zeros(len(x))
for i in range(0, len(x)):
    pdf_vector[i] = TruncNormal_pdf(x[i], a, b, mu, sigma)
plt.plot(x, pdf_vector)
plt.show()
