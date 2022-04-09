from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)
from ...utils import Utils


class Ebm(Utils):
    def ebm_reg(self, xs, y, *args, **kwargs):
        return ExplainableBoostingRegressor(*args, **kwargs).fit(xs, y)

    def ebm_class(self, xs, y, *args, **kwargs):
        return ExplainableBoostingClassifier(*args, **kwargs).fit(xs, y)
