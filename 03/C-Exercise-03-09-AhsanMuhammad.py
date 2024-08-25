# Authored by: Ahsan Muhammad (1183091)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

# All the formulaes are taken from pg.16 of lecture notes

# from H.W sheet using the hints
def truncated_normal_pdf(x, mu, sigma, a, b):
    # numerator of the trunc norm pdf - using norm.pdf in hints of the question
    num = norm.pdf(x, mu, sigma)
    # denominator of the trunc norm pdf - using norm,cdf in hints of the question
    den = sigma * (norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma))
    return num / den

def Sample_TruncNormal_AR(mu, sigma, a, b, N):
    # proposed density U[a, b] - H.W assignment says can only sample from ~ U
    proposal_density = 1 / (b - a) 
    
    x_vals = np.linspace(a, b, 1000)
    # known pdf of truncated density 
    f_x = truncated_normal_pdf(x_vals, mu, sigma, a, b)
    # c = supp f(x)/c.g(x)
    c = np.max(f_x) / proposal_density
    
    # empty list init to store accepted draes
    samples = []
    while len(samples) < N:
        # draw from uniform g(x) with a, b given
        x_proposed = np.random.uniform(a, b)
        # draw again from a uniform[0, 1] which is independent of g(x0)
        u = np.random.uniform(0, 1)
        # getting pdf
        f_x = truncated_normal_pdf(x_proposed, mu, sigma, a, b)
        # accept bound as defined - accept if u <= f(x)/c*f(x) else continue
        if u <= f_x / (c * proposal_density):
            samples.append(x_proposed)

    return samples


# Parameters
a = 0
b = 2
mu = 0.5
sigma = 1
N = 10000

# Generate samples from TruncNormal density using Accept/Reject method
samples = Sample_TruncNormal_AR(mu, sigma, a, b, N)

# Plot histogram of accedted samples
plt.hist(samples, bins=40, density=True, alpha=0.5, color='g', label='Accepted Sample')

x = np.linspace(a, b, 1000)
# generate truncated normal density
y = truncated_normal_pdf(x, mu, sigma, a, b)
plt.plot(x, y, 'r-', lw=1, label='f(x)')

plt.xlabel('x')
plt.ylabel('Density')
plt.title('Truncated Normal Distribution')
plt.legend()
plt.show()
