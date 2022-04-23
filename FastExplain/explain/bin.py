import numpy as np
from scipy.stats import t


def quantile_ied(x_vec, q):
    """
    https://github.com/DanaJomar/PyALE
    Inverse of empirical distribution function (quantile R type 1).

    More details in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
    https://en.wikipedia.org/wiki/Quantile

    Arguments:
    x_vec -- A pandas series containing the values to compute the quantile for
    q -- An array of probabilities (values between 0 and 1)
    """

    x_vec = x_vec.sort_values()
    n = len(x_vec) - 1
    m = 0
    j = (n * q + m).astype(int)  # location of the value
    g = n * q + m - j

    gamma = (g != 0).astype(int)
    quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[j] + gamma * x_vec.iloc[
        j
    ]
    quant_res.index = q
    # add min at quantile zero and max at quantile one (if needed)
    if 0 in q:
        quant_res.loc[0] = x_vec.min()
    if 1 in q:
        quant_res.loc[1] = x_vec.max()
    return quant_res


def get_bins(x, grid_size):
    """
    https://github.com/DanaJomar/PyALE
    Return quantiled bin values with number of quantiles equal to 'grid_size'
    """
    quantiles = np.append(0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size))
    bins = [x.min()] + quantile_ied(x, quantiles).to_list()
    return np.unique(bins)


def CI_estimate(x_vec, C=0.95):
    """
    https://github.com/DanaJomar/PyALE
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
