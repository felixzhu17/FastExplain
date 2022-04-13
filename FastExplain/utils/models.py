from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)


def is_xgb(m):
    return isinstance(
        m,
        (
            XGBClassifier,
            XGBRegressor,
        ),
    )


def is_rf(m):
    return isinstance(
        m,
        (
            RandomForestRegressor,
            RandomForestClassifier,
        ),
    )


def is_ebm(m):
    return isinstance(
        m,
        (
            ExplainableBoostingRegressor,
            ExplainableBoostingClassifier,
        ),
    )
