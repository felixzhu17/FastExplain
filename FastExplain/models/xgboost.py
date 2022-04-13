from xgboost import XGBRegressor, XGBClassifier
from FastExplain.metrics import m_cross_entropy, m_rmse
from FastExplain.models.hypertuning import hypertune_model
from sklearn.utils.class_weight import compute_sample_weight
import warnings


XGB_DEFAULT_PARAMS = {
    "max_depth": (3, 18, True),
    "gamma": (0, 10, False),
    "colsample_bytree": (0.3, 1, False),
    "colsample_bylevel": (0.3, 1, False),
    "learning_rate": (0.05, 0.3, False),
    "subsample": (0.1, 1, False),
    "min_child_weight": (0.01, 1, False),
}


def xgb_reg(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    max_evals=100,
    hypertune_params=XGB_DEFAULT_PARAMS,
    *args,
    **kwargs
):
    if hypertune:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_hyperparams = hypertune_model(
                xs=xs,
                y=y,
                val_xs=val_xs,
                val_y=val_y,
                model_fit_func=_xgb_reg,
                loss_metric=m_rmse,
                max_evals=max_evals,
                hypertune_params=hypertune_params,
            )
        return _xgb_reg(xs, y, **best_hyperparams)
    else:
        return _xgb_reg(xs, y, *args, **kwargs)


def xgb_class(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    max_evals=100,
    hypertune_params=XGB_DEFAULT_PARAMS,
    *args,
    **kwargs
):
    if hypertune:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_hyperparams = hypertune_model(
                xs=xs,
                y=y,
                val_xs=val_xs,
                val_y=val_y,
                model_fit_func=_xgb_class,
                loss_metric=m_cross_entropy,
                max_evals=max_evals,
                hypertune_params=hypertune_params,
            )
        return _xgb_class(xs, y, **best_hyperparams)
    else:
        return _xgb_class(xs, y, *args, **kwargs)


def _xgb_reg(
    xs,
    y,
    tree_method="hist",
    use_label_encoder=False,
    verbosity=0,
    silent=True,
    *args,
    **kwargs
):
    return XGBRegressor(
        tree_method=tree_method,
        use_label_encoder=use_label_encoder,
        verbosity=verbosity,
        silent=silent,
        *args,
        **kwargs
    ).fit(xs, y)


def _xgb_class(
    xs,
    y,
    tree_method="hist",
    use_label_encoder=False,
    verbosity=0,
    silent=True,
    sample_weight=None,
    *args,
    **kwargs
):
    sample_weight = (
        sample_weight if sample_weight else compute_sample_weight("balanced", y)
    )
    return XGBClassifier(
        tree_method=tree_method,
        use_label_encoder=use_label_encoder,
        sample_weight=sample_weight,
        verbosity=verbosity,
        silent=silent,
        *args,
        **kwargs
    ).fit(xs, y)
