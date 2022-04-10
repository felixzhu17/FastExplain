import math


def r_mse(pred, y):
    return round(math.sqrt(((pred - y) ** 2).mean()), 6)


def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)
