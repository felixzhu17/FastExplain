from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from modelflow.utils import check_numeric


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
    if _check_param_input(param_info):
        lower, upper, integer = param_info
        if integer:
            return hp.randint(param, lower, upper)
        else:
            return hp.uniform(param, lower, upper)
    else:
        raise ValueError(
            f"Key of {param} must be an array of [lower (numeric), upper (numeric), is_integer (bool)]"
        )


def _get_space(params):
    return {
        param: _get_param_range(param, param_info)
        for param, param_info in params.items()
    }


def _check_param_input(param_info):
    return (
        len(param_info) == 3
        and check_numeric(param_info[0])
        and check_numeric(param_info[1])
        and isinstance(param_info[2], bool)
    )
