import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff
from .bin import get_bins
from ..utils import *
import warnings


def get_one_way_analysis(
    df,
    x_col,
    y_col,
    grid_size=20,
    bins=None,
    dp=2,
    func=None,
    size_cutoff=0,
    percentage=False,
    condense_last=True,
):

    bins = bins if bins else get_bins(df[x_col], grid_size)
    filtered_df = df[[x_col, y_col]].copy()
    filtered_df[x_col] = pd.cut(filtered_df[x_col], bins, include_lowest=True)
    func = func if func else lambda x: conditional_mean(x, size_cutoff)
    one_way_df = filtered_df.groupby(x_col).agg(
        **{y_col: (y_col, func), "size": (y_col, "count")}
    )
    one_way_df.index = bin_intervals(one_way_df.index, dp, percentage, condense_last)
    return one_way_df


def plot_one_way_analysis(
    df, x_col, y_col, feature_names=None, plotsize=None, *args, **kwargs
):
    output = get_one_way_analysis(df, x_col, y_col, *args, **kwargs)
    return plot_one_way(
        df=output,
        cols=[x_col, y_col],
        size=output["size"],
        feature_names=feature_names,
        plotsize=plotsize,
    )


def get_two_way_analysis(
    df,
    x_cols,
    y_col,
    grid_size=20,
    bins=None,
    dp=2,
    func=None,
    size_cutoff=0,
    percentage=False,
    condense_last=True,
):
    col_1, col_2 = x_cols
    if bins:
        if len(bins) != 2:
            raise ValueError("Need two sets of bins to get two-way analysis")
        bin_1 = bins[0]
        bin_2 = bins[1]
    else:
        bin_1 = get_bins(df[col_1], grid_size)
        bin_2 = get_bins(df[col_2], grid_size)

    filtered_df = df[x_cols + [y_col]].copy()
    filtered_df[col_1] = pd.cut(filtered_df[col_1], bin_1, include_lowest=True)
    filtered_df[col_2] = pd.cut(filtered_df[col_2], bin_2, include_lowest=True)

    func = func if func else lambda x: conditional_mean(x, size_cutoff)

    two_way_df = (
        filtered_df.groupby(x_cols)
        .apply(func)
        .reset_index()
        .pivot(index=col_1, columns=col_2)[y_col]
    )

    two_way_df.index = bin_intervals(two_way_df.index, dp, percentage, condense_last)
    two_way_df.columns = bin_intervals(
        two_way_df.columns, dp, percentage, condense_last
    )
    return two_way_df


def get_two_way_frequency(df, x_cols, *args, **kwargs):

    mod_df = df.copy()
    mod_df["dummy_count"] = 1

    output = get_two_way_analysis(
        df=mod_df, x_cols=x_cols, y_col="dummy_count", func=sum, *args, **kwargs
    )

    return output / output.sum(axis=0)


def plot_two_way_frequency(
    df,
    x_cols,
    feature_names=None,
    plotsize=None,
    colorscale="Blues",
    *args,
    **kwargs,
):
    output = get_two_way_frequency(df, x_cols, *args, **kwargs)
    return plot_two_way(output, x_cols, feature_names, plotsize, colorscale)


def plot_two_way_analysis(
    df,
    x_cols,
    y_col,
    grid_size=20,
    bins=None,
    format_numbers=True,
    dp=2,
    func=None,
    feature_names=None,
    plotsize=None,
    colorscale="Blues",
    size_cutoff=0,
    percentage=False,
    condense_last=True,
):
    two_way_df = get_two_way_analysis(
        df=df,
        x_cols=x_cols,
        y_col=y_col,
        grid_size=grid_size,
        bins=bins,
        format_numbers=format_numbers,
        dp=dp,
        func=func,
        size_cutoff=size_cutoff,
        percentage=percentage,
        condense_last=condense_last,
    )
    return plot_two_way(two_way_df, x_cols, feature_names, plotsize, colorscale)


def feature_correlation(xs, plotsize=(1000, 1000)):
    keep_cols = [i for i in xs.columns if len(xs[i].unique()) > 1]
    corr = np.round(spearmanr(xs[keep_cols]).correlation, 4)
    fig = ff.create_dendrogram(
        1 - corr,
        orientation="left",
        labels=xs.columns,
        distfun=squareform,
        linkagefun=lambda x: sch.linkage(x, "average"),
    )
    fig.update_layout(width=plotsize[0], height=plotsize[1], plot_bgcolor="white")
    return fig


class OneWay:
    def __init__(self, m, xs, df=None, dep_var=None):
        self.m = m
        self.xs = xs
        self.df = df
        self.dep_var = dep_var

        if self.df is None or self.dep_var is None:
            warnings.warn(
                "One way analysis and Two way analysis does not work without dependent variable"
            )

    def get_one_way_analysis(self, x_col, *args, **kwargs):
        return get_one_way_analysis(
            self.df, x_col=x_col, y_col=self.dep_var, *args, **kwargs
        )

    def plot_one_way_analysis(self, x_col, *args, **kwargs):
        return plot_one_way_analysis(
            self.df, x_col=x_col, y_col=self.dep_var, *args, **kwargs
        )

    def get_two_way_analysis(self, x_cols, *args, **kwargs):
        return get_two_way_analysis(
            self.df, x_cols=x_cols, y_col=self.dep_var, *args, **kwargs
        )

    def plot_two_way_analysis(self, x_cols, *args, **kwargs):
        return plot_two_way_analysis(
            self.df, x_cols=x_cols, y_col=self.dep_var, *args, **kwargs
        )

    def get_two_way_frequency(self, *args, **kwargs):
        if self.df is None:
            return get_two_way_frequency(self.xs, *args, **kwargs)
        else:
            return get_two_way_frequency(self.df, *args, **kwargs)

    def plot_two_way_frequency(self, *args, **kwargs):
        if self.df is None:
            return plot_two_way_frequency(self.xs, *args, **kwargs)
        else:
            return plot_two_way_frequency(self.df, *args, **kwargs)

    def feature_correlation(self, *args, **kwargs):
        return feature_correlation(self.xs, *args, **kwargs)
