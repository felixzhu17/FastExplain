import math


def r_mse(pred, y, mean=True):
    if mean:
        return math.sqrt(((pred - y) ** 2).mean())
    else:
        return (pred - y) ** 2


def m_rmse(m, xs, y, mean=True):
    return r_mse(m.predict(xs), y, mean)
