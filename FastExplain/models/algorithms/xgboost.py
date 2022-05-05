import warnings

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from FastExplain.metrics import m_cross_entropy, m_rmse
from FastExplain.models.algorithms.hypertuning import hypertune_model
from FastExplain.models.algorithms.params import XGB_DEFAULT_HYPERTUNE_PARAMS


def xgb_reg(
    xs,
    y,
    val_xs=None,
    val_y=None,
    hypertune=False,
    hypertune_max_evals=100,
    hypertune_params=XGB_DEFAULT_HYPERTUNE_PARAMS,
    *args,
    **kwargs
):
    if hypertune:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_hyperparams, trials = hypertune_model(
                xs=xs,
                y=y,
                val_xs=val_xs,
                val_y=val_y,
                model_fit_func=_xgb_reg,
                hypertune_loss_metric=m_rmse,
                hypertune_max_evals=hypertune_max_evals,
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
    hypertune_max_evals=100,
    hypertune_params=XGB_DEFAULT_HYPERTUNE_PARAMS,
    *args,
    **kwargs
):
    if hypertune:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_hyperparams, trials = hypertune_model(
                xs=xs,
                y=y,
                val_xs=val_xs,
                val_y=val_y,
                model_fit_func=_xgb_class,
                hypertune_loss_metric=m_cross_entropy,
                hypertune_max_evals=hypertune_max_evals,
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
