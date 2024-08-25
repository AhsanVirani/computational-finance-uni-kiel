# Authored By: Ahsan Muhammad (1183091)
# Dated: 02.05.2024
# Group 12

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def log_returns(data) -> pd.Series:
    """
    Calculates the log returns of the given data

    Args:
        data (Series): price series

    Returns:
        pd.Series: series of log returns
    """
    return np.log(data) - np.log(data.shift(1))

def annualized_empiral_mean(data):
    """
    Calculates the annualized empirical mean from log return data

    Args:
        data (Series): log returns series

    Returns:
        pd.Series: series of annualized empirical mean
    """
    return (250 / (len(data) - 1)) * data.sum()
    
def annualized_empircal_sd(data) -> pd.Series:
    """
    Calculates the standard deviation of the given data

    Args:
        data (Series): log returns series

    Returns:
        pd.Series: series of standard deviation
    """
    u_hat = annualized_empiral_mean(data)
    variance = (250 / (len(data) - 2)) * ((data - (u_hat / 250)) ** 2).sum()
    return np.sqrt(variance)

##########################################################################
# Part a - Load data and calculate log return
##########################################################################
data = pd.read_csv("time_series_dax_2024.csv", delimiter=";", index_col="Date").sort_index()
# Calculate log return
lr = log_returns(data['Close'])
# calculate empirical mean
annualized_mu = annualized_empiral_mean(lr)
# calculate empirical standard deviation
annualized_sd = annualized_empircal_sd(lr)

##########################################################################
# Part b - Plotting
##########################################################################
lr.plot()
plt.xlabel('Date', fontsize=12)  
plt.ylabel('Log Returns', fontsize=12)  
plt.title('Log Returns of DAX Index', fontsize=12)  
plt.axhline(annualized_mu, color='green', linestyle='--', alpha=0.5)
plt.axhline(annualized_sd, color='red', linestyle='--', alpha=0.5)

##########################################################################
# Part C - Simulation
##########################################################################
log_return_sample = np.random.normal(loc=annualized_mu, scale=annualized_sd, size=len(data))
print(annualized_mu, annualized_sd)
log_return_sample = pd.Series(data=log_return_sample, index=data.index)
print(log_return_sample)
log_return_sample.plot(color='grey', alpha=0.5)

plt.legend(loc='upper right', labels=['Log Return', 'Annualized Empirical Mean', 'Annualized Empirical SD', 'Simulated Log Return'])
plt.show()


##########################################################################
# Part d
# The spikes of the simulated data are much bigger in both directions, underlying the even split between up and down moves
# For the log returns empirical data, we see stylized facts about the time series - such that higher period of volatility are followed
# by higher periods of volatility and lower periods of volatility by lower periods of volatility. This is known as volatility clustering, 
# which the simulated data fails to capture. 
##########################################################################
