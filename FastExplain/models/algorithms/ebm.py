from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)


def ebm_reg(xs, y, val_xs=None, val_y=None, max_bins=20, *args, **kwargs):
    return ExplainableBoostingRegressor(max_bins=max_bins, *args, **kwargs).fit(xs, y)


def ebm_class(xs, y, val_xs=None, val_y=None, max_bins=20, *args, **kwargs):
    return ExplainableBoostingClassifier(max_bins=max_bins, *args, **kwargs).fit(xs, y)
