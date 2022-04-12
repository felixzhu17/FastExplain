import pandas as pd
from modelflow.clean.base import Clean


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


class FillMissing(Clean):
    def __init__(self, fill_strategy="median", fill_const=0, na_dummy=True):
        self.cont_names = None
        self.fill_strategy = self._assign_fill_strategy(fill_strategy, fill_const)
        self.na_dummy = na_dummy

    def fit(self, xs, cont_names):
        self.cont_names = cont_names
        self.replace_xs = xs[cont_names].apply(self.fill_strategy)
        missing_xs = pd.isnull(xs[self.cont_names])
        self.missing_cols = list(missing_xs.any()[missing_xs.any()].keys())
        if self.na_dummy:
            self.na_dummy_cols = [n + "_na" for n in self.missing_cols]
            self.na_dummy_col_mappings = {}

    def fit_transform(self, xs, cont_names):
        self.fit(xs, cont_names)
        return self.transform(xs)

    def transform(self, xs):
        missing_xs = pd.isnull(xs[self.cont_names])
        if self.na_dummy:
            for n in self.missing_cols:
                missing_cat = missing_xs[n].astype("category")
                xs.loc[:, n + "_na"] = missing_cat.cat.codes
                self.na_dummy_col_mappings[n + "_na"] = dict(
                    enumerate(missing_cat.cat.categories)
                )

        xs[self.cont_names] = xs[self.cont_names].fillna(self.replace_xs)
        return xs

    def _assign_fill_strategy(self, fill_strategy, fill_const):
        if callable(fill_strategy):
            return fill_strategy
        else:
            if fill_strategy not in dir(FillStrategy):
                raise ValueError("Fill strategy not valid")
            return getattr(FillStrategy(fill_const), fill_strategy)
