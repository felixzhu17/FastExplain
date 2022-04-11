import pandas as pd
import plotly.express as px
from sklearn.inspection import partial_dependence
import plotly.graph_objects as go
from ..utils import *


def plot_pdp(m, xs, col, log_x=False, plotsize=None, *args, **kwargs):
    part_dep = partial_dependence(m, xs, [col], kind="average", *args, **kwargs)
    average = part_dep["average"]
    values = part_dep["values"]
    fig = px.line(x=values[0], y=average.squeeze(), log_x=log_x)
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_multi_pdp(m, xs, cols, index, plotsize=None, *args, **kwargs):
    pdp = {
        i: fill_list(
            list(
                partial_dependence(m, xs, [i], kind="average", *args, **kwargs)[
                    "average"
                ].squeeze()
            ),
            len(index),
        )
        for i in cols
    }

    pdp_df = pd.DataFrame(pdp, index=index)
    fig = px.line(pdp_df, x=pdp_df.index, y=pdp_df.columns)
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_ice(
    m,
    xs,
    col,
    feature_name=None,
    dep_name=None,
    plotsize=None,
    *args,
    **kwargs,
):
    ice = partial_dependence(m, xs, col, kind="individual", *args, **kwargs)
    ice_df = pd.DataFrame(ice["individual"][0]).T
    ice_df.index = ice["values"][0]
    average_effect = ice_df.mean(axis=1)
    fig = px.line(ice_df, x=ice_df.index, y=ice_df.columns)
    fig.update_traces(line_color="#456987")
    fig.add_trace(
        go.Scatter(
            x=ice_df.index,
            y=average_effect,
            name="Average",
            line=dict(color="royalblue"),
        )
    )
    fig.update_traces(showlegend=False, plot_bgcolor="white")

    feature_name = feature_name if feature_name else clean_text(col)

    fig.update_layout(title=f"ICE Plot of {feature_name}", xaxis_title=feature_name)
    if dep_name:
        fig.update_layout(yaxis_title=clean_text(dep_name))
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    return fig


class PDP:
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    def plot_pdp(self, *args, **kwargs):
        return plot_pdp(self.m, self.xs, *args, **kwargs)

    def plot_multi_pdp(self, *args, **kwargs):
        return plot_multi_pdp(self.m, self.xs, *args, **kwargs)

    def plot_ice(self, *args, **kwargs):
        return plot_ice(self.m, self.xs, *args, **kwargs)
