RF_DEFAULT_HYPERTUNE_PARAMS = {
    "n_estimators": (40, 200, True),
    "max_features": (0.1, 1, False),
    "min_samples_leaf": (0.001, 0.1, False),
    "min_samples_split": (0.001, 0.1, False),
}

XGB_DEFAULT_HYPERTUNE_PARAMS = {
    "max_depth": (3, 18, True),
    "gamma": (0, 10, False),
    "colsample_bytree": (0.3, 1, False),
    "colsample_bylevel": (0.3, 1, False),
    "learning_rate": (0.05, 0.3, False),
    "subsample": (0.1, 1, False),
    "min_child_weight": (0.01, 1, False),
}


def get_default_hypertune_params(model):
    if model == "rf":
        return RF_DEFAULT_HYPERTUNE_PARAMS
    elif model == "xgb":
        return XGB_DEFAULT_HYPERTUNE_PARAMS
    else:
        return None
