from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .logic import clean_text
from .colours import COLOURS


def plot_two_way(df, cols, feature_names=None, plotsize=None, colorscale="Blues"):
    fig = px.imshow(df, color_continuous_scale=colorscale)

    feature_1, feature_2 = feature_names if feature_names else clean_text(
        cols[0]
    ), clean_text(cols[1])

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
    if size is not None:
        fig = create_secondary_axis_plotly(px.line(df, x=df.index, y=cols[1]))
        fig.add_trace(
            go.Bar(
                name="Frequency", x=df.index, y=size, marker={"color": COLOURS["grey"]}
            ),
            secondary_y=False,
        )
        _two_axis_layout(fig)
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
    else:
        fig = px.line(df, x=df.index, y=cols[1])
    feature_1, feature_2 = feature_names if feature_names else clean_text(
        cols[0]
    ), clean_text(cols[1])
    fig.update_layout(
        title=f"{feature_1} vs {feature_2}",
        xaxis_title=feature_1,
        yaxis_title=feature_2,
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
    color = color if color else COLOURS["blue"]
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
    traces, x, size, x_axis_title=None, y_axis_title=None, plotsize=None
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


def create_secondary_axis_plotly(fig):
    output = make_subplots(specs=[[{"secondary_y": True}]])
    traces = get_plotly_express_traces(fig)
    for i in traces:
        output.add_trace(i, secondary_y=True)
    return output


def custom_legend_name(fig, new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
    return
