import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from functools import reduce
from .logic import ifnone
import re
from sklearn.model_selection import train_test_split


def quantile_ied(x_vec, q):
    """
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

    quantiles = np.append(0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size))
    bins = [x.min()] + quantile_ied(x, quantiles).to_list()
    return np.unique(bins)


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


def get_train_val_split_index(df, perc_train, seed=0, stratify=None):
    return train_test_split(
        range(len(df)),
        test_size=1 - perc_train,
        random_state=seed,
        stratify=stratify,
    )


def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int16, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint16, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }
    if obj2cat:
        typemap[
            "object"
        ] = "category"  # User wants to categorify dtype('Object'), which may not always save space
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()),
                None,
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes


def df_shrink(df, skip=[], obj2cat=False, int2uint=False):
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    return df.astype(dt)


def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub("[Dd]ate$", "", field_name))
    attr = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    if time:
        attr = attr + ["Hour", "Minute", "Second"]
    # Pandas removed `dt.week` in v1.1.10
    week = (
        field.dt.isocalendar().week.astype(field.dt.day.dtype)
        if hasattr(field.dt, "isocalendar")
        else field.dt.week
    )
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower()) if n != "Week" else week
    mask = ~field.isna()
    df[prefix + "Elapsed"] = np.where(
        mask, field.values.astype(np.int64) // 10 ** 9, np.nan
    )
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def _get_elapsed(df, field_names, date_field, base_field, prefix):
    for f in field_names:
        day1 = np.timedelta64(1, "D")
        last_date, last_base, res = np.datetime64(), None, []
        for b, v, d in zip(df[base_field].values, df[f].values, df[date_field].values):
            if last_base is None or b != last_base:
                last_date, last_base = np.datetime64(), b
            if v:
                last_date = d
            res.append(((d - last_date).astype("timedelta64[D]") / day1))
        df[prefix + f] = res
    return df


def add_elapsed_times(df, field_names, date_field, base_field):
    "Add in `df` for each event in `field_names` the elapsed time according to `date_field` grouped by `base_field`"
    # Make sure date_field is a date and base_field a bool
    df[field_names] = df[field_names].astype("bool")
    make_date(df, date_field)

    work_df = df[field_names + [date_field, base_field]]
    work_df = work_df.sort_values([base_field, date_field])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, "After")
    work_df = work_df.sort_values([base_field, date_field], ascending=[True, False])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, "Before")

    for a in ["After" + f for f in field_names] + ["Before" + f for f in field_names]:
        work_df[a] = work_df[a].fillna(0).astype(int)

    for a, s in zip([True, False], ["_bw", "_fw"]):
        work_df = work_df.set_index(date_field)
        tmp = (
            work_df[[base_field] + field_names]
            .sort_index(ascending=a)
            .groupby(base_field)
            .rolling(7, min_periods=1)
            .sum()
        )
        if base_field in tmp:
            tmp.drop(base_field, axis=1, inplace=True)
        tmp.reset_index(inplace=True)
        work_df.reset_index(inplace=True)
        work_df = work_df.merge(tmp, "left", [date_field, base_field], suffixes=["", s])
    work_df.drop(field_names, axis=1, inplace=True)
    return df.merge(work_df, "left", [date_field, base_field])


def cont_cat_split(dfs, max_card=20, dep_var=""):
    "Helper function that returns column names of cont and cat variables from given `df`."
    df = dfs.copy()
    cont_names, cat_names = [], []
    for label in df.columns:
        if label in list(dep_var):
            continue
        if (
            pd.api.types.is_integer_dtype(df[label].dtype)
            and df[label].unique().shape[0] > max_card
        ) or pd.api.types.is_float_dtype(df[label].dtype):
            cont_names.append(label)
        else:
            cat_names.append(label)
    if dep_var in cont_names:
        cont_names.remove(dep_var)
    if dep_var in cat_names:
        cat_names.remove(dep_var)
    return cont_names, cat_names
