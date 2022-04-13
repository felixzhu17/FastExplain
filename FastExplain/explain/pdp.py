import pandas as pd
import plotly.express as px
from sklearn.inspection import partial_dependence
import plotly.graph_objects as go
from FastExplain.utils import clean_text, COLOURS


def plot_ice(
    m,
    xs,
    col,
    feature_name=None,
    dep_name=None,
    plotsize=None,
    filter=None,
    sample=500,
    *args,
    **kwargs,
):
    xs = xs.query(filter) if filter else xs.sample(sample)
    ice = partial_dependence(m, xs, col, kind="individual", *args, **kwargs)
    ice_df = pd.DataFrame(ice["individual"][0]).T
    ice_df.index = ice["values"][0]
    average_effect = ice_df.mean(axis=1)
    fig = go.Figure(
        [
            go.Scatter(
                x=ice_df.index,
                y=ice_df[i],
                name=None,
                line=dict(color=COLOURS["grey"]),
            )
            for i in ice_df.columns
        ]
        + [
            go.Scatter(
                x=ice_df.index,
                y=average_effect,
                name="Average",
                line=dict(color=COLOURS["blue"]),
            )
        ]
    )
    fig.update_traces(showlegend=False)
    feature_name = feature_name if feature_name else clean_text(col)
    fig.update_layout(
        title=f"ICE Plot of {feature_name}",
        xaxis_title=feature_name,
        plot_bgcolor="white",
    )
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

    def plot_ice(self, *args, **kwargs):
        return plot_ice(self.m, self.xs, *args, **kwargs)
