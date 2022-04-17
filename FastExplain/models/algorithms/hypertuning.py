from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from FastExplain.utils import check_numeric, is_rf, is_xgb, is_ebm


def hypertune_model(
    xs,
    y,
    val_xs,
    val_y,
    model_fit_func,
    loss_metric,
    max_evals,
    hypertune_params,
):
    _check_param_dict(hypertune_params)
    space = _get_space(hypertune_params)

    def objective(space):
        m = model_fit_func(xs, y, **space)
        evaluation = [(xs, y), (val_xs, val_y)]
        metric = loss_metric(m, val_xs, val_y)
        return {"loss": metric, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False,
    )
    return best_hyperparams


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
        raise TypeError
