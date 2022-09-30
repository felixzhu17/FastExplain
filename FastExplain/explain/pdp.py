from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import partial_dependence

from FastExplain.utils import COLOURS, clean_text, doc_setter, ifnone


def plot_ice(
    m: type,
    xs: pd.DataFrame,
    col: str,
    sample: int = 500,
    filter: Optional[str] = None,
    dep_name: Optional[str] = None,
    feature_name: Optional[str] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    *pdp_args,
    **pdp_kwargs,
):

    """
    Plot ICE values for a predictor feature in a model

    Args:
        m (type):
            Trained model that uses feature as a predictor.
        xs (pd.DataFrame):
            Dataframe used by model to predict.
        col (str):
            Name of predictor feature to use for ICE
        sample (int):
            Maximum number of samples to use for ICE
        filter (Optional[str], optional):
            The query string to evaluate.
            You can refer to variables
            in the environment by prefixing them with an '@' character like
            ``@a + b``.
            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuations (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "list", "for", "import", etc) cannot be used.
            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.
            For more information refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html.
            Defaults to None.
        dep_name (Optional[str], optional):
            Custom name to use for dependent variable on plot. Defaults to None.
        feature_name (Optional[str], optional):
            Custom name to use for predictor variable on plot. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        *pdp_args, **pdp_kwargs:
            Additional arguments for SKlearn partial dependence function. See https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
    """

    xs = xs.query(filter).iloc[:sample] if filter else xs.sample(sample)
    ice = partial_dependence(m, xs, col, kind="individual", *pdp_args, **pdp_kwargs)
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
    feature_name = ifnone(feature_name, clean_text(col))
    temp_title = (
        f"ICE Plot of {feature_name} vs {clean_text(dep_name)}"
        if dep_name
        else f"ICE Plot of {feature_name}"
    )
    title = ifnone(title, temp_title)
    fig.update_layout(
        title=title,
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
    """Connected interface for PDP methods. Intended for usage with full model pipeline class. (FastExplain.models.base)"""

    def __init__(self, m, xs, dep_var=None):
        self.m = m
        self.xs = xs
        self.dep_var = dep_var

    @doc_setter(plot_ice)
    def plot_ice(
        self,
        col: str,
        sample: int = 500,
        filter: Optional[str] = None,
        dep_name: Optional[str] = None,
        feature_name: Optional[str] = None,
        title: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        *pdp_args,
        **pdp_kwargs,
    ):

        dep_name = ifnone(dep_name, self.dep_var)
        return plot_ice(
            m=self.m,
            xs=self.xs,
            col=col,
            sample=sample,
            filter=filter,
            dep_name=dep_name,
            feature_name=feature_name,
            title=title,
            plotsize=plotsize,
            *pdp_args,
            **pdp_kwargs,
        )
