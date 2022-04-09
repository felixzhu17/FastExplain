from xgboost import XGBRegressor, XGBClassifier
from ...utils import Utils


class XGBoost(Utils):
    def xgb_reg(self, xs, y, tree_method="hist", *args, **kwargs):
        return XGBRegressor(tree_method=tree_method, *args, **kwargs).fit(xs, y)

    def xgb_class(self, xs, y, tree_method="hist", *args, **kwargs):
        return XGBClassifier(tree_method=tree_method, *args, **kwargs).fit(xs, y)
