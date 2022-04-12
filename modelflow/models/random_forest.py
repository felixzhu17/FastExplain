from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from modelflow.metrics import m_cross_entropy, m_rmse
from modelflow.models.hypertuning import hypertune_model

RF_DEFAULT_PARAMS = {
    "n_estimators": (40, 200, True),
    "max_features": (0.1, 1, False),
    "min_samples_leaf": (0.001, 0.1, False),
    "min_samples_split": (0.001, 0.1, False),
}


def rf_reg(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    max_evals=100,
    hypertune_params=RF_DEFAULT_PARAMS,
    n_estimators=40,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    if hypertune:
        best_hyperparams = hypertune_model(
            xs=xs,
            y=y,
            val_xs=val_xs,
            val_y=val_y,
            model_fit_func=_rf_reg,
            loss_metric=m_rmse,
            max_evals=max_evals,
            hypertune_params=hypertune_params,
        )
        return _rf_reg(xs, y, **best_hyperparams)
    else:
        return _rf_reg(
            xs,
            y,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            *args,
            **kwargs,
        )


def rf_class(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    max_evals=100,
    hypertune_params=RF_DEFAULT_PARAMS,
    n_estimators=40,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    if hypertune:
        best_hyperparams = hypertune_model(
            xs=xs,
            y=y,
            val_xs=val_xs,
            val_y=val_y,
            model_fit_func=_rf_class,
            loss_metric=m_cross_entropy,
            max_evals=max_evals,
            hypertune_params=hypertune_params,
        )
        return _rf_class(xs, y, **best_hyperparams)
    else:
        return _rf_class(
            xs,
            y,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            *args,
            **kwargs,
        )


def _rf_reg(
    xs,
    y,
    max_samples=200_000,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestRegressor(
        n_jobs=-1,
        max_samples=max_samples,
        oob_score=True,
        *args,
        **kwargs,
    ).fit(xs, y)


def _rf_class(
    xs,
    y,
    max_samples=200_000,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestClassifier(
        n_jobs=-1,
        max_samples=max_samples,
        class_weight="balanced",
        oob_score=True,
        *args,
        **kwargs,
    ).fit(xs, y)
