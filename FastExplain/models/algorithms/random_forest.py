from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from FastExplain.metrics import m_cross_entropy, m_rmse
from FastExplain.models.algorithms.hypertuning import hypertune_model
from FastExplain.models.algorithms.params import RF_DEFAULT_HYPERTUNE_PARAMS


def rf_reg(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    hypertune_max_evals=100,
    hypertune_params=RF_DEFAULT_HYPERTUNE_PARAMS,
    *args,
    **kwargs,
):
    if hypertune:
        best_hyperparams, trials = hypertune_model(
            xs=xs,
            y=y,
            val_xs=val_xs,
            val_y=val_y,
            model_fit_func=_rf_reg,
            hypertune_loss_metric=m_rmse,
            hypertune_max_evals=hypertune_max_evals,
            hypertune_params=hypertune_params,
        )
        return _rf_reg(xs, y, **best_hyperparams)
    else:
        return _rf_reg(
            xs,
            y,
            *args,
            **kwargs,
        )


def rf_class(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    hypertune_max_evals=100,
    hypertune_params=RF_DEFAULT_HYPERTUNE_PARAMS,
    *args,
    **kwargs,
):
    if hypertune:
        best_hyperparams, trials = hypertune_model(
            xs=xs,
            y=y,
            val_xs=val_xs,
            val_y=val_y,
            model_fit_func=_rf_class,
            hypertune_loss_metric=m_cross_entropy,
            hypertune_max_evals=hypertune_max_evals,
            hypertune_params=hypertune_params,
        )
        return _rf_class(xs, y, **best_hyperparams)
    else:
        return _rf_class(
            xs,
            y,
            *args,
            **kwargs,
        )


def _rf_reg(
    xs,
    y,
    n_jobs=-1,
    max_samples=200_000,
    n_estimators=40,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestRegressor(
        n_jobs=n_jobs,
        max_samples=max_samples,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        *args,
        **kwargs,
    ).fit(xs, y)


def _rf_class(
    xs,
    y,
    n_jobs=-1,
    max_samples=200_000,
    class_weight="balanced",
    n_estimators=40,
    max_features=0.5,
    min_samples_leaf=5,
    *args,
    **kwargs,
):
    max_samples = min(len(xs), max_samples)
    return RandomForestClassifier(
        n_jobs=n_jobs,
        max_samples=max_samples,
        class_weight=class_weight,
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        *args,
        **kwargs,
    ).fit(xs, y)
