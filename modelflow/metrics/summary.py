from modelflow.explain.one_way import get_one_way_analysis, get_two_way_analysis
from modelflow.utils import plot_one_way, plot_two_way


def get_benchmark_error(func, benchmark, train_y, val_y, y, mean):
    return {
        "train": func(benchmark, train_y, mean),
        "val": func(benchmark, val_y, mean),
        "overall": func(benchmark, y, mean),
    }


def get_error(func, m, train_xs, train_y, val_xs, val_y, xs, y, mean):
    return {
        "train": func(m, train_xs, train_y, mean),
        "val": func(m, val_xs, val_y, mean),
        "overall": func(m, xs, y, mean),
    }


def get_one_way_error(df, error, x_col, *args, **kwargs):
    error_df = df.copy()
    error_df["error"] = error
    return get_one_way_analysis(error_df, x_col, "error", *args, **kwargs)


def plot_one_way_error(
    df, error, x_col, feature_names=None, plotsize=None, *args, **kwargs
):
    output = get_one_way_error(df, error, x_col, *args, **kwargs)
    return plot_one_way(
        output,
        [x_col, "error"],
        size=output["size"],
        feature_names=feature_names,
        plotsize=plotsize,
        *args,
        **kwargs,
    )


def get_two_way_error(df, error, x_cols, *args, **kwargs):
    error_df = df.copy()
    error_df["error"] = error
    return get_two_way_analysis(error_df, x_cols, "error", *args, **kwargs)


def plot_two_way_error(
    df,
    error,
    x_cols,
    feature_names=None,
    plotsize=None,
    colorscale="Blues",
    *args,
    **kwargs,
):
    output = get_two_way_error(df, error, x_cols, *args, **kwargs)
    return plot_two_way(output, x_cols, feature_names, plotsize, colorscale)
