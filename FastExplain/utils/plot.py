import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from FastExplain.utils.colours import COLOURS, CORE_COLOURS
from FastExplain.utils.logic import check_list_type, clean_text, ifnone


def plot_two_way(
    df,
    x_cols,
    feature_names=None,
    dep_name=None,
    title=None,
    plotsize=None,
    colorscale="Blues",
    surface_plot=True,
):
    """Base function for plotting two-way analysis"""
    if surface_plot:
        return _plot_two_way_surface(
            df=df,
            x_cols=x_cols,
            feature_names=feature_names,
            dep_name=dep_name,
            title=title,
            plotsize=plotsize,
            colorscale=colorscale,
        )

    else:
        return _plot_two_way_heatmap(
            df=df,
            x_cols=x_cols,
            feature_names=feature_names,
            dep_name=dep_name,
            title=title,
            plotsize=plotsize,
            colorscale=colorscale,
        )


def _plot_two_way_surface(
    df,
    x_cols,
    feature_names=None,
    dep_name=None,
    title=None,
    plotsize=None,
    colorscale="Blues",
):
    """Base function for plotting two-way surface plot"""

    X_AXIS_LABELS = list(df.columns)
    X_AXIS_VALS = list(range(len(X_AXIS_LABELS)))

    Y_AXIS_LABELS = list(df.index)
    Y_AXIS_VALS = list(range(len(Y_AXIS_LABELS)))

    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(x_cols[0]), clean_text(x_cols[1]))
    )

    title = ifnone(
        title,
        f"{feature_1} and {feature_2} vs {dep_name}"
        if dep_name
        else f"{feature_1} and {feature_2}",
    )

    fig = go.Figure(data=[go.Surface(z=df, colorscale=colorscale)])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title_text=feature_2,
                ticktext=X_AXIS_LABELS,
                tickvals=X_AXIS_VALS,
            ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title_text=feature_1,
                ticktext=Y_AXIS_LABELS,
                tickvals=Y_AXIS_VALS,
            ),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title_text=ifnone(dep_name, "Target"),
            ),
        ),
    )

    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )

    return fig


def _plot_two_way_heatmap(
    df,
    x_cols,
    feature_names=None,
    dep_name=None,
    title=None,
    plotsize=None,
    colorscale="Blues",
):
    """Base function for plotting two-way heatmap"""
    fig = px.imshow(df, color_continuous_scale=colorscale)

    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(x_cols[0]), clean_text(x_cols[1]))
    )

    title = ifnone(
        title,
        f"{feature_1} and {feature_2} vs {dep_name}"
        if dep_name
        else f"{feature_1} and {feature_2}",
    )

    fig = plotly_layout(
        fig,
        plotsize=plotsize,
        title=title,
        xaxis_title=feature_2,
        yaxis_title=feature_1,
    )
    return fig


def plot_bar(
    df,
    x_col,
    y_col,
    xaxis_title=None,
    yaxis_title=None,
    title=None,
    plotsize=None,
    sort=False,
    ascending=True,
    color=COLOURS["grey"],
):
    """Base function for plotting frequency"""
    xaxis_title = ifnone(xaxis_title, clean_text(x_col))
    yaxis_title = ifnone(yaxis_title, clean_text(y_col))
    df = sort_plot_df(df, y_col, sort, ascending)
    fig = go.Figure(
        data=[
            go.Bar(
                name=yaxis_title,
                x=df.index,
                y=df[y_col],
                marker={"color": color},
            )
        ]
    )
    title = ifnone(title, f"{xaxis_title} vs {yaxis_title}")
    fig = plotly_layout(
        fig,
        plotsize=plotsize,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    return fig


def plot_one_way(
    df,
    x_col,
    y_col,
    size=None,
    xaxis_title=None,
    yaxis_title=None,
    title=None,
    plotsize=None,
    sort=False,
    ascending=True,
    display_proportion=False,
    histogram_name=None,
):
    """Base function for plotting one-way analysis with frequency"""

    xaxis_title = ifnone(xaxis_title, clean_text(x_col))

    yaxis_title = ifnone(
        yaxis_title,
        clean_text(", ".join(y_col)) if check_list_type(y_col) else clean_text(y_col),
    )
    title = ifnone(title, f"{xaxis_title} vs {yaxis_title}")
    histogram_name = ifnone(
        histogram_name, "Proportion" if display_proportion else "Frequency"
    )

    table_cols = [i for i in df.columns if i != "size"]
    plot_colours = CORE_COLOURS[: len(table_cols)]

    clean_col_map = {i: clean_text(i) for i in table_cols}
    clean_col_names = list(clean_col_map.values())
    df.rename(columns=clean_col_map, inplace=True)

    if size is not None:

        if display_proportion:
            size = size / size.sum()

        df["size"] = size

        df = sort_plot_df(df, clean_col_names[0], sort, ascending)

        fig = create_secondary_axis_plotly(
            px.line(
                df, x=df.index, y=clean_col_names, color_discrete_sequence=plot_colours
            )
        )

        fig.add_trace(
            go.Bar(
                name=histogram_name,
                x=df.index,
                y=df["size"],
                marker={"color": COLOURS["grey"]},
            ),
            secondary_y=False,
        )

        fig = plotly_two_axis_layout(
            fig,
            x_axis_title=xaxis_title,
            primary_y_axis_title=histogram_name,
            secondary_y_axis_title=yaxis_title,
            title=title,
            plotsize=plotsize,
        )

    else:
        df = sort_plot_df(df, clean_col_names[0], sort, ascending)
        fig = px.line(
            df, x=df.index, y=clean_col_names, color_discrete_sequence=plot_colours
        )
        fig = plotly_layout(
            fig,
            plotsize=plotsize,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )

    return fig


def plot_two_one_way(
    df, x_col, y_col, xaxis_title=None, yaxis_title=None, title=None, plotsize=None
):

    xaxis_title = ifnone(xaxis_title, clean_text(x_col))
    yaxis_title_1, yaxis_title_2 = ifnone(
        yaxis_title, (clean_text(y_col[0]), clean_text(y_col[1]))
    )
    title = ifnone(title, f"{xaxis_title} vs {yaxis_title_1} and {yaxis_title_2}")

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

    fig = plotly_two_axis_layout(
        fig,
        x_axis_title=xaxis_title,
        primary_y_axis_title=yaxis_title_2,
        secondary_y_axis_title=yaxis_title_1,
        title=title,
        plotsize=plotsize,
    )
    fig["data"][0]["name"] = yaxis_title_1
    fig["data"][0]["showlegend"] = True
    fig["data"][1]["name"] = yaxis_title_2
    fig["data"][1]["showlegend"] = True
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
    title=None,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i in traces:
        fig.add_trace(i, secondary_y=True)
    fig.add_trace(
        go.Bar(name="Frequency", x=x, y=size, marker={"color": COLOURS["grey"]}),
        secondary_y=False,
    )
    fig = plotly_two_axis_layout(
        fig,
        x_axis_title=x_axis_title,
        primary_y_axis_title="Frequency",
        secondary_y_axis_title=clean_text(y_axis_title),
        title=title,
        plotsize=plotsize,
    )

    return fig


def plotly_layout(fig, plotsize, line_colours=None, legend_names=None, *args, **kwargs):
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        legend_title_text="",
        plot_bgcolor="white",
        *args,
        **kwargs,
    )
    if line_colours is not None:
        adjust_line_colour(fig, line_colours)
    if legend_names is not None:
        custom_legend_name(fig, legend_names)
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


def plotly_two_axis_layout(
    fig,
    x_axis_title,
    primary_y_axis_title,
    secondary_y_axis_title,
    plotsize,
    *args,
    **kwargs,
):
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
        )
    )

    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=primary_y_axis_title, secondary_y=False)
    fig.update_yaxes(title_text=secondary_y_axis_title, secondary_y=True)
    fig = plotly_layout(fig, plotsize, *args, **kwargs)

    return fig


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


def create_two_axis_plot(fig_1, fig_2):
    fig = create_secondary_axis_plotly(fig_1)
    traces = get_plotly_express_traces(fig_2)
    for i in traces:
        fig.add_trace(i, secondary_y=False)
    return fig


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


def append_traces(fig, new_fig):
    traces = get_plotly_express_traces(new_fig)
    for i in traces:
        fig.add_trace(i)
    return fig


def adjust_line_colour(fig, colours):
    for i, colour in enumerate(colours):
        fig.data[i].line.color = colour
    return


def custom_legend_name(fig, new_names):
    for i, new_name in enumerate(new_names):
        fig.data[i].name = new_name
        fig["data"][i]["showlegend"] = True
    return
