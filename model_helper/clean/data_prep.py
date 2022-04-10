import pandas as pd
import numpy as np
import re


def prepare_data(
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


class PandasClean:
    def __init__(
        self,
        df,
        cat_names=None,
        cont_names=None,
        dep_var=None,
        perc_train=0.8,
        seed=0,
        splits=None,
        cat_strategy="ordinal",
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
