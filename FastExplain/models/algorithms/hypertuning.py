from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from FastExplain.clean import check_classification
from FastExplain.metrics import m_cross_entropy, m_rmse
from FastExplain.utils import check_numeric, ifnone, is_ebm, is_rf, is_xgb


class Hypertune:
    def __init__(
        self,
        xs,
        y,
        val_xs,
        val_y,
        model_fit_func,
        hypertune_loss_metric=None,
    ):

        self.xs = xs
        self.y = y
        self.val_xs = val_xs
        self.val_y = val_y
        self.model_fit_func = (
            prepare_model_class(model_fit_func)
            if isinstance(model_fit_func, type)
            else model_fit_func
        )
        self.hypertune_loss_metric = ifnone(
            hypertune_loss_metric,
            m_cross_entropy if check_classification(self.y) else m_rmse,
        )

    def hypertune_model(
        self, hypertune_params, hypertune_max_evals=100, *model_args, **model_kwargs
    ):
        self.hypertune_params = hypertune_params
        self.hypertune_max_evals = hypertune_max_evals
        self.best_hyperparams, self.trials = hypertune_model(
            xs=self.xs,
            y=self.y,
            val_xs=self.val_xs,
            val_y=self.val_y,
            model_fit_func=self.model_fit_func,
            hypertune_loss_metric=self.hypertune_loss_metric,
            hypertune_max_evals=hypertune_max_evals,
            hypertune_params=hypertune_params,
            *model_args,
            **model_kwargs,
        )
        self.m = self.model_fit_func(
            self.xs, self.y, *model_args, **model_kwargs, **self.best_hyperparams
        )
        return self.m


def prepare_model_class(model_fit_func):
    _check_model_fit_func(model_fit_func)

    def func(xs, y, *args, **kwargs):
        return model_fit_func(*args, **kwargs).fit(xs, y)

    return func


def _check_model_fit_func(model_fit_func: type):
    if hasattr(model_fit_func, "fit") and hasattr(model_fit_func, "predict"):
        return
    else:
        raise ValueError(f"{model_fit_func} needs to have fit and predict attributes")


def hypertune_model(
    xs,
    y,
    val_xs,
    val_y,
    model_fit_func,
    hypertune_loss_metric,
    hypertune_max_evals,
    hypertune_params,
    *model_args,
    **model_kwargs,
):
    _check_param_dict(hypertune_params)
    space = _get_space(hypertune_params)

    def objective(space):
        m = model_fit_func(xs, y, *model_args, **space, **model_kwargs)
        evaluation = [(xs, y), (val_xs, val_y)]
        metric = hypertune_loss_metric(m, val_xs, val_y)
        return {"loss": metric, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=hypertune_max_evals,
        trials=trials,
        verbose=False,
    )
    trials = sorted(trials.trials, key=lambda x: x["result"]["loss"])
    return best_hyperparams, trials


def _get_param_range(param, param_info):
    lower, upper, integer = param_info
    if integer:
        return hp.randint(param, lower, upper)
    else:
        return hp.uniform(param, lower, upper)


def _get_space(params):
    return {
        param: _get_param_range(param, param_info)
        for param, param_info in params.items()
    }


def _check_param_dict(params):
    if not isinstance(params, dict):
        raise ValueError(
            f"Params must be supplied in a dictionary format (hypertune_params)"
        )
    param_checks = {k: _check_param_input(v) for k, v in params.items()}
    param_checked = [k for k, v in param_checks.items() if v is False]
    if len(param_checked) > 0:
        raise ValueError(
            f"Params of {' '.join(param_checked)} must be an array of [lower (numeric), upper (numeric), is_integer (bool)]"
        )
    else:
        return


def _check_param_input(param_info):
    return (
        len(param_info) == 3
        and check_numeric(param_info[0])
        and check_numeric(param_info[1])
        and isinstance(param_info[2], bool)
    )


def get_model_parameters(m):
    if is_rf(m):
        return {i: getattr(m, i) for i in m._get_param_names()}
    elif is_xgb(m):
        return m.get_xgb_params()
    elif is_ebm(m):
        return m.get_params()
    else:
        return "Model not supported"
