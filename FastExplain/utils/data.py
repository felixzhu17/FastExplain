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


def trim_df(df, bins):
    if bins < 0:
        df = df.iloc[-bins:]
    else:
        df = df.iloc[:-bins]

    return df


def fill_categorical_nan(x, value="NaN"):
    if x.isna().sum() > 0:
        x = x.cat.add_categories([value])
        x = x.fillna(value)
    return x


def adjust_df(df):
    adjust = -1 * df.iloc[0]["eff"]
    df["eff"] += adjust
    df["lower"] += adjust
    df["upper"] += adjust
    return df


def get_range_first_value(index):
    return [float(i.replace("+", "").split(" - ")[0]) for i in index]


def condense_interpretable_df(df):
    def get_range_first_value(index):
        return [float(str(i).replace("+", "").split(" - ")[0]) for i in index]

    condensed_df = df.groupby(df.index).apply(
        lambda x: pd.Series(
            [
                np.average(x["eff"], weights=x["size"])
                if sum(x["size"]) > 0
                else np.average(x["eff"]),
                np.average(x["upper"], weights=x["size"])
                if sum(x["size"]) > 0
                else np.average(x["upper"]),
                np.average(x["lower"], weights=x["size"])
                if sum(x["size"]) > 0
                else np.average(x["lower"]),
                sum(x["size"]),
            ],
            index=["eff", "upper", "lower", "size"],
        )
    )

    return condensed_df.sort_index(key=get_range_first_value)
