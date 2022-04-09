from .random_forest import RandomForest
from .xgboost import XGBoost
from .ebm import Ebm


class Models(RandomForest, XGBoost, Ebm):
    ...
