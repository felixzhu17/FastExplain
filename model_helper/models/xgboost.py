from xgboost import XGBRegressor, XGBClassifier


def xgb_reg(xs, y, tree_method="hist", *args, **kwargs):
    return XGBRegressor(tree_method=tree_method, *args, **kwargs).fit(xs, y)


def xgb_class(xs, y, tree_method="hist", *args, **kwargs):
    return XGBClassifier(tree_method=tree_method, *args, **kwargs).fit(xs, y)
