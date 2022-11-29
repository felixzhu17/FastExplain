import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from FastExplain.utils import ifnone


def get_train_val_split_index(df, perc_train, seed=0, stratify=None):
    if perc_train == 0:
        return list(range(len(df))), []
    else:
        return train_test_split(
            range(len(df)),
            test_size=1 - perc_train,
            random_state=seed,
            stratify=stratify,
        )


def split_train_val(xs, y, splits):
    if y is None:
        return xs.loc[splits[0]], xs.loc[splits[1]]
    else:
        return xs.loc[splits[0]], y.loc[splits[0]], xs.loc[splits[1]], y.loc[splits[1]]


def cont_cat_split(dfs, max_card=20, max_sparsity=0.25, dep_var=None, verbose=True):
    "Helper function that returns column names of cont and cat variables from given `df`."
    dep_var = ifnone(dep_var, "")
    df = dfs.copy()
    cont_names, cat_names = [], []
    for label in df.columns:
        if check_unique(df[label]):
            if verbose:
                warnings.warn(
                    f"There is only {df[label].unique().shape[0]} unique value of {label}. This is too few to be included as a model feature."
                )
            continue
        if check_cont_col(df[label], max_card=max_card):
            cont_names.append(label)
        else:
            if check_sparsity(df[label], max_sparsity=max_sparsity):
                if verbose:
                    warnings.warn(
                        f"There are {df[label].unique().shape[0]} unique values of {label}. This is too many to be included as a model feature."
                    )
                continue
            else:
                cat_names.append(label)
    if dep_var in cont_names:
        cont_names.remove(dep_var)
    if dep_var in cat_names:
        cat_names.remove(dep_var)
    return cont_names, cat_names


def check_cont_col(x, max_card=20):
    return (
        pd.api.types.is_integer_dtype(x.dtype) or pd.api.types.is_float_dtype(x.dtype)
    ) and x.unique().shape[0] > max_card


def check_sparsity(x, max_sparsity=0.25):
    return x.unique().shape[0] > max_sparsity * x.shape[0]


def check_unique(x):
    try:
        return len(x.unique()) == 1
    except TypeError as e:
        return False
