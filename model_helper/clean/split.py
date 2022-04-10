import pandas as pd
from sklearn.model_selection import train_test_split


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
    return xs.loc[splits[0]], y.loc[splits[0]], xs.loc[splits[1]], y.loc[splits[1]]


def cont_cat_split(dfs, max_card=20, dep_var=""):
    "Helper function that returns column names of cont and cat variables from given `df`."
    df = dfs.copy()
    cont_names, cat_names = [], []
    for label in df.columns:
        if label in list(dep_var):
            continue
        if check_cont_col(df[label], max_card=max_card):
            cont_names.append(label)
        else:
            cat_names.append(label)
    if dep_var in cont_names:
        cont_names.remove(dep_var)
    if dep_var in cat_names:
        cat_names.remove(dep_var)
    return cont_names, cat_names


def check_cont_col(x, max_card=20):
    return (
        pd.api.types.is_integer_dtype(x.dtype) and x.unique().shape[0] > max_card
    ) or pd.api.types.is_float_dtype(x.dtype)
