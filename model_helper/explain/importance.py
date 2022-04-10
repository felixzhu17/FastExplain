import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import interpret
from ..utils import *


def plot_feature_importance(m, xs, feature_highlights=[], limit=10, plotsize=None):

    df = _get_feature_importance_df(m, xs)
    df = df.sort_values("Importance")
    df = df[-1 * limit :]
    plot_dict = {"x": "Importance", "y": "Feature", "orientation": "h"}

    if "Error" in df.columns:
        plot_dict["error_x"] = "Error"

    fig = px.bar(df, **plot_dict)

    colour = df.apply(
        lambda x: COLOURS["blue"]
        if x["Feature"] in feature_highlights
        else COLOURS["grey"],
        axis=1,
    )

    fig.update_traces(marker=dict(color=colour))
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    fig.update_layout(plot_bgcolor="white")

    return fig


def _get_feature_importance_df(m, xs):
    rf = isinstance(
        m,
        (
            sklearn.ensemble._forest.RandomForestRegressor,
            sklearn.ensemble._forest.RandomForestClassifier,
        ),
    )

    ebm = isinstance(
        m,
        (
            interpret.glassbox.ebm.ebm.ExplainableBoostingRegressor,
            interpret.glassbox.ebm.ebm.ExplainableBoostingClassifier,
        ),
    )

    if ebm:
        df_dict = {
            "Feature": m.explain_global().data()["names"],
            "Importance": m.explain_global().data()["scores"],
        }
    else:
        df_dict = {
            "Feature": xs.columns,
            "Importance": m.feature_importances_,
        }

    if rf:
        df_dict["Error"] = np.std(
            [tree.feature_importances_ for tree in m.estimators_], axis=0
        )

    return pd.DataFrame(df_dict)


class Importance:
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    def plot_feature_importance(self, *args, **kwargs):
        return plot_feature_importance(self.m, self.xs, *args, **kwargs)
