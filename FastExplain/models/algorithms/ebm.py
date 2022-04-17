from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)


def ebm_reg(xs, y, val_xs=None, val_y=None, *args, **kwargs):
    return ExplainableBoostingRegressor(*args, **kwargs).fit(xs, y)


def ebm_class(xs, y, val_xs=None, val_y=None, *args, **kwargs):
    return ExplainableBoostingClassifier(*args, **kwargs).fit(xs, y)
