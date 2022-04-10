import pandas as pd
from .base import Clean


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

    def fit_transform(self, xs, cont_names):
        self.fit(xs, cont_names)
        return self.transform(xs, cont_names)

    def transform(self, xs):
        missing_xs = pd.isnull(xs[self.cont_names])
        missing_cols = missing_xs.any()[missing_xs.any()].keys()
        if self.na_dummy:
            for n in missing_cols:
                xs.loc[:, n + "_na"] = missing_xs[n].astype("category").cat.codes
        xs[self.cont_names] = xs[self.cont_names].fillna(self.replace_xs)
        return xs

    def _assign_fill_strategy(self, fill_strategy, fill_const):
        if callable(fill_strategy):
            return fill_strategy
        else:
            if fill_strategy not in dir(FillStrategy):
                raise ValueError("Fill strategy not valid")
            return getattr(FillStrategy(fill_const), fill_strategy)
