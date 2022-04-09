import pandas as pd
import numpy as np
import re
from ..utils import Utils
from sklearn.model_selection import train_test_split


class DataPrepUtils(Utils):
    def get_train_val_split_index(self, df, perc_train, seed=0, stratify=None):
        return train_test_split(
            range(len(df)),
            test_size=1 - perc_train,
            random_state=seed,
            stratify=stratify,
        )

    def df_shrink_dtypes(self, df, skip=[], obj2cat=True, int2uint=False):
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

    def df_shrink(self, df, skip=[], obj2cat=False, int2uint=False):
        "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
        dt = self.df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
        return df.astype(dt)

    def make_date(self, df, date_field):
        "Make sure `df[date_field]` is of the right date type."
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

    def add_datepart(self, df, field_name, prefix=None, drop=True, time=False):
        "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
        self.make_date(df, field_name)
        field = df[field_name]
        prefix = self.ifnone(prefix, re.sub("[Dd]ate$", "", field_name))
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
            mask, field.values.astype(np.int64) // 10**9, np.nan
        )
        if drop:
            df.drop(field_name, axis=1, inplace=True)
        return df

    def _get_elapsed(self, df, field_names, date_field, base_field, prefix):
        for f in field_names:
            day1 = np.timedelta64(1, "D")
            last_date, last_base, res = np.datetime64(), None, []
            for b, v, d in zip(
                df[base_field].values, df[f].values, df[date_field].values
            ):
                if last_base is None or b != last_base:
                    last_date, last_base = np.datetime64(), b
                if v:
                    last_date = d
                res.append(((d - last_date).astype("timedelta64[D]") / day1))
            df[prefix + f] = res
        return df

    def add_elapsed_times(self, df, field_names, date_field, base_field):
        "Add in `df` for each event in `field_names` the elapsed time according to `date_field` grouped by `base_field`"
        # Make sure date_field is a date and base_field a bool
        df[field_names] = df[field_names].astype("bool")
        self.make_date(df, date_field)

        work_df = df[field_names + [date_field, base_field]]
        work_df = work_df.sort_values([base_field, date_field])
        work_df = self._get_elapsed(
            work_df, field_names, date_field, base_field, "After"
        )
        work_df = work_df.sort_values([base_field, date_field], ascending=[True, False])
        work_df = self._get_elapsed(
            work_df, field_names, date_field, base_field, "Before"
        )

        for a in ["After" + f for f in field_names] + [
            "Before" + f for f in field_names
        ]:
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
            work_df = work_df.merge(
                tmp, "left", [date_field, base_field], suffixes=["", s]
            )
        work_df.drop(field_names, axis=1, inplace=True)
        return df.merge(work_df, "left", [date_field, base_field])

    def cont_cat_split(self, dfs, max_card=20, dep_var=""):
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


class FillStrategy:
    "Namespace containing the various filling strategies."

    def __init__(self, fill):
        self.fill = fill

    def median(self, c):
        return c.median()

    def constant(self, c):
        return self.fill

    def mode(self, c):
        return c.dropna().value_counts().idxmax()


class DataPrep(DataPrepUtils):
    def prepare_data(
        self,
        df,
        cat_names=None,
        cont_names=None,
        dep_var=None,
        perc_train=0.8,
        seed=0,
        splits=None,
        normalize=False,
        one_hot=False,
        fill_strategy="median",
        fill_const=0,
        na_dummy=True,
        reduce_memory=True,
        return_class=False,
    ):
        pandas_clean = PandasClean(
            df=df,
            cat_names=cat_names,
            cont_names=cont_names,
            dep_var=dep_var,
            perc_train=perc_train,
            seed=seed,
            splits=splits,
            normalize=normalize,
            one_hot=one_hot,
            fill_strategy=fill_strategy,
            fill_const=fill_const,
            na_dummy=na_dummy,
            reduce_memory=reduce_memory,
        )

        if return_class:
            return pandas_clean

        else:
            if perc_train == 0:
                return (
                    pandas_clean.train_xs,
                    pandas_clean.train_y,
                )
            else:
                return (
                    pandas_clean.train_xs,
                    pandas_clean.train_y,
                    pandas_clean.val_xs,
                    pandas_clean.val_y,
                )


class PandasClean(DataPrepUtils):
    def __init__(
        self,
        df,
        cat_names=None,
        cont_names=None,
        dep_var=None,
        perc_train=0.8,
        seed=0,
        splits=None,
        normalize=False,
        one_hot=False,
        fill_strategy="median",
        fill_const=0,
        na_dummy=True,
        reduce_memory=True,
    ):
        self.df = df.reset_index(drop=True)
        self.cat_names = cat_names.copy()
        self.cont_names = cont_names.copy()
        self.dep_var = dep_var
        self.perc_train = perc_train
        self.splits = splits
        self.seed = seed
        self.one_hot = one_hot
        self._one_hot_cat_names = []
        self.cat_mapping = {}
        self._assign_fill_strategy(fill_strategy, fill_const)
        self.na_dummy = na_dummy
        if reduce_memory:
            self.df = self.df_shrink(self.df, int2uint=True)

        # Check categorical y
        if self.dep_var:
            self._check_categorical()

        # Get splits
        if self.splits is None:
            self._get_splits()

        # Convert column types
        self._prepare_categorical_cols()

        # Gather data
        self.xs = self.df[self.cat_names + self.cont_names]
        if self.dep_var:
            self.y = self.df[self.dep_var]

        # Perform splits
        self._split_train_val()

        # Fill Missing
        self._fill_missing()

        # Optional Normalize
        if normalize:
            self._normalize()

        # Gather final data
        self.xs = pd.concat([self.train_xs, self.val_xs])
        if self.dep_var:
            self.y = pd.concat([self.train_y, self.val_y])

        self.df = self.xs.assign(**{dep_var: self.y})

    def _assign_fill_strategy(self, fill_strategy, fill_const):
        if callable(fill_strategy):
            self.fill_strategy = fill_strategy
        else:
            if fill_strategy not in dir(FillStrategy):
                raise ValueError("Fill strategy not valid")
            self.fill_strategy = getattr(FillStrategy(fill_const), fill_strategy)

    def _check_categorical(self):
        unique_dep_var = len(self.df[self.dep_var].unique())
        if unique_dep_var == 1:
            raise ValueError("Dependent Variable only has 1 unique value")
        self.categorical = unique_dep_var == 2
        if self.categorical:
            y_cat = self.df[self.dep_var].astype("category")
            self.cat_mapping[self.dep_var] = dict(enumerate(y_cat.cat.categories))
            self.df[self.dep_var] = y_cat.cat.codes
        else:
            self.df[self.dep_var] = self.df[self.dep_var].astype("float")

    def _get_splits(self):
        if self.perc_train == 0:
            self.splits = list(range(len(self.df))), []
        else:
            self.splits = self.get_train_val_split_index(
                self.df,
                self.perc_train,
                seed=self.seed,
                stratify=self.df[self.dep_var] if self.categorical else None,
            )

    def _prepare_categorical_cols(self):
        for cat_col in self.cat_names:
            if self.one_hot:
                self._one_hot_column(cat_col)
            else:
                self._ordinal_encode_column(cat_col)
        if self.one_hot:
            self.cat_names = self._one_hot_cat_names

    def _split_train_val(self):
        self.train_xs, self.val_xs = (
            self.xs.loc[self.splits[0]],
            self.xs.loc[self.splits[1]],
        )
        self.train_y, self.val_y = (
            self.y.loc[self.splits[0]],
            self.y.loc[self.splits[1]],
        )

    def _fill_missing(self):
        replace_xs = self.train_xs[self.cont_names].apply(self.fill_strategy)
        missing_train = pd.isnull(self.train_xs[self.cont_names])
        missing_val = pd.isnull(self.val_xs[self.cont_names])
        missin_train_keys = missing_train.any()[missing_train.any()].keys()
        if self.na_dummy:
            for n in missin_train_keys:
                cat_convert = missing_train[n].astype("category")
                self.train_xs.loc[:, n + "_na"] = cat_convert.cat.codes
                self.cat_mapping[n + "_na"] = dict(
                    enumerate(cat_convert.cat.categories)
                )
                self.val_xs.loc[:, n + "_na"] = (
                    missing_val[n].astype("category").cat.codes
                )
        self.train_xs[self.cont_names] = self.train_xs[self.cont_names].fillna(
            replace_xs
        )
        self.val_xs[self.cont_names] = self.val_xs[self.cont_names].fillna(replace_xs)

    def _normalize(self):
        mean_xs = self.train_xs[self.cont_names].mean()
        std_xs = self.train_xs[self.cont_names].std()
        self.train_xs[self.cont_names] = (
            self.train_xs[self.cont_names] - mean_xs
        ) / std_xs
        self.val_xs[self.cont_names] = (self.val_xs[self.cont_names] - mean_xs) / std_xs

    def _one_hot_column(self, cat_col):
        one_hot = pd.get_dummies(self.df[cat_col])
        one_hot.columns = [f"{cat_col}_{i}" for i in one_hot.columns]
        self.df = self.df.drop(cat_col, axis=1)
        self.df = self.df.join(one_hot)
        self._one_hot_cat_names.extend(list(one_hot.columns))
        return self.df

    def _ordinal_encode_column(self, cat_col):
        if pd.isnull(self.df[cat_col]).any():
            self.df[cat_col] = self.df[cat_col].fillna("None")
        cat_convert = self.df[cat_col].astype("category")
        self.cat_mapping[cat_col] = dict(enumerate(cat_convert.cat.categories))
        self.df[cat_col] = cat_convert.cat.codes
