import numpy as np
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt

# trunc normal pdf
def pdf_trunc_normal(a, b, x, mu, sigma):
    # the norm.pdf already contains the scaling factor 1/sigma
    return norm.pdf(x,  mu, sigma) / (norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)) if a <= x <= b else 0

def Sample_TruncNormal_AR(a, b, mu, sigma, N):
    # deterministic - uniform
    g_x = 1 / (b - a)
    # generate a uniform random variables on a, b with sample = 1000
    sample_size = 1000
    U = np.random.uniform(a, b, sample_size)
    f_x = [pdf_trunc_normal(a=a, b=b, x=x, mu=mu, sigma=sigma) for x in U]
    # given a = 0, b = 2 and sigma = 1, we can use the fact that mu = mode = 0.5 - we use more general solution
    C = np.max(f_x) / g_x
    
    # Go on with algorithm
    accepted_sample = []
    while len(accepted_sample) < N:
        # draw from two uniform random vars - one for generating Y other for accepting / rejecting
        U = np.random.uniform(0, 1, 2)
        # shift U[0, 1] to U[0, b - a] then add a to make it U[a, b] - our intended pdf g_x = 1/(b - a)
        Y = U[0] * (b - a) + a
        if U[1] <= pdf_trunc_normal(a, b, x=Y, mu=mu, sigma=sigma) / (C * g_x):
            accepted_sample.append(Y)

    return accepted_sample

a = 0
b = 2
mu = 0.5
sigma = 1
N = 1000

# Generate samples
X = Sample_TruncNormal_AR(a, b, mu, sigma, N)

# Plot histogram
plt.hist(X, 50, density=True, alpha=0.6, color='g')

# Plot exact pdf
x = np.linspace(a - a / 20, b + b / 20, 1000)
pdf_vector = np.array([pdf_trunc_normal(a, b, xi, mu, sigma) for xi in x])
plt.plot(x, pdf_vector, 'r-', lw=2)
plt.title('Histogram of Sampled Truncated Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()