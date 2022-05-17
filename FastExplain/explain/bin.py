import numpy as np
from scipy.stats import t


def get_bins(x, grid_size):
    """Return quantiled bin values with number of quantiles equal to 'grid_size'"""
    return np.unique([x.quantile(i / grid_size) for i in range(grid_size + 1)])


def CI_estimate(x_vec, C=0.95):
    """
    Estimate the size of the confidence interval of a data sample.

    The confidence interval of the given data sample (x_vec) is
    [mean(x_vec) - returned value, mean(x_vec) + returned value].
    """
    alpha = 1 - C
    n = len(x_vec)
    stand_err = x_vec.std() / np.sqrt(n)
    critical_val = 1 - (alpha / 2)
    z_star = stand_err * t.ppf(critical_val, n - 1)
    return z_star
