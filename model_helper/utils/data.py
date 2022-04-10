import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from functools import reduce
from .logic import ifnone
import re
from sklearn.model_selection import train_test_split


def percent_cat_agg(series, top=None):
    x = Counter(series)
    sum_x = sum(x.values())
    x_perc = {k: v / sum_x for k, v in x.items()}
    if top:
        return Counter(x_perc).most_common(top)
    else:
        return x_perc


def encode_list(df, col, include_col_name=True):
    mlb = MultiLabelBinarizer()
    df = df.join(
        pd.DataFrame(
            mlb.fit_transform(df[col]),
            columns=[f"{col}_{i}" for i in mlb.classes_]
            if include_col_name
            else mlb.classes_,
            index=df.index,
        )
    )
    return df


def merge_multi_df(dfs, *args, **kwargs):
    return reduce(lambda left, right: pd.merge(left, right, *args, **kwargs), dfs)


def get_date_freq(df, col, freq):
    return df.groupby(df[col].dt.to_period(freq)).count()[col]
