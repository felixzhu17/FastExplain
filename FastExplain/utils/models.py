from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


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
