"""Date cleaning functions taken from FastAI https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py"""

import re

import numpy as np
import pandas as pd

from FastExplain.utils import ifnone


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
    week = (
        field.dt.isocalendar().week.astype(field.dt.day.dtype)
        if hasattr(field.dt, "isocalendar")
        else field.dt.week
    )
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower()) if n != "Week" else week
    mask = ~field.isna()
    df[prefix + "Elapsed"] = np.where(
        mask, field.values.astype(np.int64) // 10**9, np.nan
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
