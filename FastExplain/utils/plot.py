import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from FastExplain.utils.colours import COLOURS
from FastExplain.utils.logic import clean_text, ifnone


def plot_two_way(
    df,
    x_cols,
    feature_names=None,
    dep_name=None,
    plot_title=None,
    plotsize=None,
    colorscale="Blues",
):
    """Base function for plotting two-way analysis"""
    fig = px.imshow(df, color_continuous_scale=colorscale)

    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(x_cols[0]), clean_text(x_cols[1]))
    )

    plot_title = ifnone(
        plot_title,
        f"{feature_1} and {feature_2} vs {dep_name}"
        if dep_name
        else f"{feature_1} and {feature_2}",
    )
    fig.update_layout(
        title=plot_title,
        xaxis_title=feature_2,
        yaxis_title=feature_1,
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


# def plot_frequency(df, x_col, size, x_axis_name=None,
#     y_axis_name=None,
#     plot_title=None,
#     plotsize=None,
#     sort=None,
#     ascending=True,):
#     df["size"] = size
#     df = sort_plot_df(df, y_col, sort, ascending)


def plot_one_way(
    df,
    x_col,
    y_col,
    size=None,
    x_axis_name=None,
    y_axis_name=None,
    plot_title=None,
    plotsize=None,
    sort=None,
    ascending=True,
):
    """Base function for plotting one-way analysis with frequency"""

    x_axis_name = ifnone(x_axis_name, clean_text(x_col))
    y_axis_name = ifnone(y_axis_name, clean_text(y_col))

    if size is not None:
        df["size"] = size
        df = sort_plot_df(df, y_col, sort, ascending)
        fig = create_secondary_axis_plotly(
            px.line(df, x=df.index, y=y_col, color_discrete_sequence=[COLOURS["blue"]])
        )
        fig.add_trace(
            go.Bar(
                name="Frequency",
                x=df.index,
                y=df["size"],
                marker={"color": COLOURS["grey"]},
            ),
            secondary_y=False,
        )
        _two_axis_layout(fig)
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
        fig.update_yaxes(title_text=y_axis_name, secondary_y=True)
    else:
        df = sort_plot_df(df, y_col, sort, ascending)
        fig = px.line(df, x=df.index, y=y_col)
        fig.update_yaxes(title_text=y_axis_name)

    plot_title = ifnone(plot_title, f"{x_axis_name} vs {y_axis_name}")

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_axis_name,
        plot_bgcolor="white",
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


def plot_two_one_way(
    df, x_col, y_col, x_axis_name=None, y_axis_name=None, plot_title=None, plotsize=None
):

    x_axis_name = ifnone(x_axis_name, clean_text(x_col))
    y_axis_name_1, y_axis_name_2 = ifnone(
        y_axis_name, (clean_text(y_col[0]), clean_text(y_col[1]))
    )

    fig = create_secondary_axis_plotly(
        px.line(df, x=df.index, y=y_col[0], color_discrete_sequence=[COLOURS["blue"]])
    )
    [
        fig.add_trace(i, secondary_y=False)
        for i in get_plotly_express_traces(
            px.line(
                df, x=df.index, y=y_col[1], color_discrete_sequence=[COLOURS["red"]]
            )
        )
    ]
    _two_axis_layout(fig)
    fig["data"][0]["name"] = y_axis_name_1
    fig["data"][0]["showlegend"] = True
    fig["data"][1]["name"] = y_axis_name_2
    fig["data"][1]["showlegend"] = True
    fig.update_yaxes(title_text=y_axis_name_2, secondary_y=False)
    fig.update_yaxes(title_text=y_axis_name_1, secondary_y=True)

    plot_title = ifnone(
        plot_title, f"{x_axis_name} vs {y_axis_name_1} and {y_axis_name_2}"
    )
    fig.update_layout(
        title=plot_title,
        xaxis_title=x_axis_name,
        plot_bgcolor="white",
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


def get_upper_lower_bound_traces(
    x,
    y,
    y_lower,
    y_upper,
    size,
    color=None,
    line_name="",
    return_index_size=True,
):
    color = ifnone(color, COLOURS["blue"])
    fig = go.Figure(
        [
            go.Scatter(
                name=clean_text(line_name),
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color),
            ),
            go.Scatter(
                name="Upper Bound",
                x=x,
                y=y_upper,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Lower Bound",
                x=x,
                y=y_lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
        ]
    )
    if return_index_size:
        return list(fig["data"]), x, size
    else:
        return list(fig["data"])


def plot_upper_lower_bound_traces(
    traces,
    x,
    size,
    x_axis_title=None,
    y_axis_title=None,
    plotsize=None,
    plot_title=None,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i in traces:
        fig.add_trace(i, secondary_y=True)
    fig.add_trace(
        go.Bar(name="Frequency", x=x, y=size, marker={"color": COLOURS["grey"]}),
        secondary_y=False,
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    _two_axis_layout(fig)
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_layout(
        title=plot_title,
    )
    if y_axis_title:
        fig.update_yaxes(title_text=clean_text(y_axis_title), secondary_y=True)
    return fig


def _two_axis_layout(fig):
    fig.update_layout(
        dict(
            yaxis2={
                "anchor": "x",
                "overlaying": "y",
                "side": "left",
                "showgrid": False,
            },
            yaxis={
                "anchor": "x",
                "domain": [0.0, 1.0],
                "side": "right",
                "showgrid": False,
            },
            xaxis={"showgrid": False},
            plot_bgcolor="white",
        )
    )
    return


def get_plotly_express_traces(fig):
    traces = []
    for trace in range(len(fig["data"])):
        traces.append(fig["data"][trace])
    return traces


def create_secondary_axis_plotly(fig, fig_on_secondary_axis=True):
    output = make_subplots(specs=[[{"secondary_y": True}]])
    traces = get_plotly_express_traces(fig)
    for i in traces:
        output.add_trace(i, secondary_y=fig_on_secondary_axis)
    return output


def custom_legend_name(fig, new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
    return


def sort_plot_df(df, y_col, sort, ascending):
    if sort in ["frequency", "Frequency"]:
        return df.sort_values("size", ascending=ascending)
    elif sort:
        return df.sort_values(y_col, ascending=ascending)
    else:
        return df
