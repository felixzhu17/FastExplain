import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from FastExplain.utils.colours import COLOURS
from FastExplain.utils.logic import clean_text, ifnone


def plot_two_way(df, cols, feature_names=None, plotsize=None, colorscale="Blues"):
    fig = px.imshow(df, color_continuous_scale=colorscale)

    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(cols[0]), clean_text(cols[1]))
    )

    fig.update_layout(
        title=f"Joint Distribution of {feature_1} and {feature_2}",
        xaxis_title=feature_2,
        yaxis_title=feature_1,
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


def plot_one_way(df, cols, size=None, feature_names=None, plotsize=None):
    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(cols[0]), clean_text(cols[1]))
    )

    if size is not None:
        fig = create_secondary_axis_plotly(
            px.line(
                df, x=df.index, y=cols[1], color_discrete_sequence=[COLOURS["blue"]]
            )
        )
        fig.add_trace(
            go.Bar(
                name="Frequency", x=df.index, y=size, marker={"color": COLOURS["grey"]}
            ),
            secondary_y=False,
        )
        _two_axis_layout(fig)
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
        fig.update_yaxes(title_text=feature_2, secondary_y=True)
    else:
        fig = px.line(df, x=df.index, y=cols[1])
        fig.update_yaxes(title_text=feature_2)

    fig.update_layout(
        title=f"{feature_1} vs {feature_2}",
        xaxis_title=feature_1,
        plot_bgcolor="white",
    )
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


def plot_two_one_way(df, cols, feature_names=None, plotsize=None):
    feature_1, feature_2, feature_3 = ifnone(
        feature_names, (clean_text(cols[0]), clean_text(cols[1]), clean_text(cols[2]))
    )

    fig = create_secondary_axis_plotly(
        px.line(df, x=df.index, y=cols[1], color_discrete_sequence=[COLOURS["blue"]])
    )
    [
        fig.add_trace(i, secondary_y=False)
        for i in get_plotly_express_traces(
            px.line(df, x=df.index, y=cols[2], color_discrete_sequence=[COLOURS["red"]])
        )
    ]
    _two_axis_layout(fig)
    fig["data"][0]["name"] = feature_2
    fig["data"][0]["showlegend"] = True
    fig["data"][1]["name"] = feature_3
    fig["data"][1]["showlegend"] = True
    fig.update_yaxes(title_text=feature_3, secondary_y=False)
    fig.update_yaxes(title_text=feature_2, secondary_y=True)

    fig.update_layout(
        title=f"{feature_1} vs {feature_2} and {feature_3}",
        xaxis_title=feature_1,
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
    main_title=None,
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
        title=main_title,
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
