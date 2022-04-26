from FastExplain.utils import root_mean


def r_mse(pred, y, mean=True):
    """Root mean squared error"""
    if mean:
        return root_mean((pred - y) ** 2)
    else:
        return (pred - y) ** 2


def m_rmse(m, xs, y, mean=True):
    """Root mean squared error for Regression model"""
    return r_mse(m.predict(xs), y, mean)
