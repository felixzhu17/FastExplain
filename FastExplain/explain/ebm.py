import pandas as pd

from FastExplain.utils import (
    COLOURS,
    bin_columns,
    clean_text,
    cycle_colours,
    get_upper_lower_bound_traces,
    ifnone,
    merge_multi_df,
    plot_upper_lower_bound_traces,
)


def ebm_explain_summary(m, xs, col, model_names=None, *args, **kwargs):
    if isinstance(m, (list, tuple)):
        model_names = ifnone(model_names, [f"Model {i}" for i in range(len(m))])
        ebms = []
        for count, ale_info in enumerate(zip(m, xs)):
            model, x_values = ale_info
            if count == len(m) - 1:
                ebms.append(
                    _clean_ebm_explain(m=model, xs=x_values, col=col, *args, **kwargs)[
                        ["eff", "size"]
                    ]
                )
            else:
                ebms.append(
                    _clean_ebm_explain(m=model, xs=x_values, col=col, *args, **kwargs)[
                        ["eff"]
                    ]
                )

        output = merge_multi_df(ebms, left_index=True, right_index=True)
        output.columns = model_names + ["size"]
        return output
    else:
        return _clean_ebm_explain(m, xs, col, *args, **kwargs)


def _clean_ebm_explain(m, xs, col, dp=2, percentage=False, condense_last=True):

    ebm_global = m.explain_global()
    index = _get_ebm_index(m, col)
    binned = pd.cut(xs[col], ebm_global.data(index)["names"], include_lowest=True)
    binned_count = list(binned.groupby(binned).count())
    df = pd.DataFrame(
        {
            "eff": ebm_global.data(index)["scores"],
            "upper": ebm_global.data(index)["upper_bounds"],
            "lower": ebm_global.data(index)["lower_bounds"],
            "size": binned_count,
        },
        index=bin_columns(
            ebm_global.data(index)["names"],
            dp=dp,
            percentage=percentage,
            condense_last=condense_last,
        ),
    )
    df = df[~df.index.duplicated(keep="last")]
    adjust = -1 * df.iloc[0]["eff"]
    df["eff"] += adjust
    df["lower"] += adjust
    df["upper"] += adjust
    return df


def plot_ebm_explain(
    m,
    xs,
    col,
    dep_name=None,
    feature_name=None,
    model_names=None,
    main_title=None,
    plotsize=None,
    *args,
    **kwargs,
):

    feature_name = ifnone(feature_name, clean_text(col))

    if isinstance(m, list):
        model_names = (
            model_names if model_names else [f"Model {i}" for i in range(len(m))]
        )
        for count, ale_info in enumerate(zip(m, xs, model_names, cycle_colours())):
            model, x_values, model_name, color = ale_info
            if count == 0:
                traces, x, size = _get_ebm_explain_traces(
                    m=model,
                    xs=x_values,
                    col=col,
                    model_name=model_name,
                    color=color,
                    return_index_size=True,
                    *args,
                    **kwargs,
                )
            else:
                traces.extend(
                    _get_ebm_explain_traces(
                        m=model,
                        xs=x_values,
                        col=col,
                        model_name=model_name,
                        color=color,
                        return_index_size=False,
                        *args,
                        **kwargs,
                    )
                )
    else:
        traces, x, size = _get_ebm_explain_traces(
            m=m,
            xs=xs,
            col=col,
            model_name=feature_name,
            color=COLOURS["blue"],
            return_index_size=True,
            *args,
            **kwargs,
        )

    title = (
        f"EBM {feature_name} vs {clean_text(dep_name)}"
        if dep_name
        else f"EBM {feature_name}"
    )
    main_title = ifnone(main_title, title)

    fig = plot_upper_lower_bound_traces(
        traces=traces,
        x=x,
        size=size,
        x_axis_title=feature_name,
        y_axis_title=dep_name,
        plotsize=plotsize,
        main_title=main_title,
    )
    return fig


def _get_ebm_explain_traces(
    m, xs, col, model_name, color, return_index_size=True, *args, **kwargs
):
    df = ebm_explain_summary(m, xs, col, *args, **kwargs)
    x = df.index
    y = df["eff"]
    size = df["size"]
    y_lower = df["lower"]
    y_upper = df["upper"]
    return get_upper_lower_bound_traces(
        x=x,
        y=y,
        y_lower=y_lower,
        y_upper=y_upper,
        size=size,
        color=color,
        line_name=model_name,
        return_index_size=return_index_size,
    )


class EbmExplain:
    def __init__(self, m, xs, dep_var=None):
        self.m = m
        self.xs = xs
        self.dep_var = dep_var

    def ebm_explain_summary(self, *args, **kwargs):
        return ebm_explain_summary(self.m, self.xs, *args, **kwargs)

    def plot_ebm_explain(self, col, dep_name=None, *args, **kwargs):
        dep_name = ifnone(dep_name, self.dep_var)
        return plot_ebm_explain(
            self.m, self.xs, col, dep_name=dep_name, *args, **kwargs
        )


def _get_ebm_index(m, col):
    ebm_global = m.explain_global()
    col_dict = {i: count for count, i in enumerate(ebm_global.data()["names"])}
    return col_dict[col]
