from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def rf_reg(
    xs,
    y,
    n_estimators=40,
    max_samples=200_000,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        *args,
        **kwargs,
    ).fit(xs, y)


def rf_class(
    xs,
    y,
    n_estimators=40,
    max_samples=200_000,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestClassifier(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        oob_score=True,
        *args,
        **kwargs,
    ).fit(xs, y)
