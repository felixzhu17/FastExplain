import math
import numbers
from collections import Counter
from functools import reduce

import numpy as np
import pandas as pd


def percent_cat_agg(series, top=None):
    x = Counter(series)
    sum_x = sum(x.values())
    x_perc = {k: v / sum_x for k, v in x.items()}
    if top:
        return Counter(x_perc).most_common(top)
    else:
        return x_perc


def merge_multi_df(dfs, *args, **kwargs):
    return reduce(lambda left, right: pd.merge(left, right, *args, **kwargs), dfs)


def conditional_mean(x, cutoff):
    if len(x) < cutoff:
        return None
    else:
        return np.mean(x)


def sample_index(df, *args, **kwargs):
    return list(df.sample(*args, **kwargs).index)


def query_df_index(df, query):
    return list(df.query(query).index)


def root_mean(x):
    return math.sqrt(np.mean(x))
