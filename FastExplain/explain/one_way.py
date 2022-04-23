import warnings
from typing import List, Optional, Callable, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from FastExplain.clean import check_cont_col
from FastExplain.explain.bin import get_bins
from FastExplain.utils import (
    bin_intervals,
    conditional_mean,
    ifnone,
    plot_one_way,
    plot_two_one_way,
    plot_two_way,
    clean_text,
)


def feature_correlation(xs: pd.DataFrame, plotsize: List[int] = (1000, 1000)):
    """
    Plot dendogram of hierarchical clustering of spearman correlation between variables

    Args:
        xs (pd.DataFrame):
            Dataframe containing variables
        plotsize (List[int], optional):
            Custom plotsize supplied as (width, height). Defaults to (1000, 1000).
    """
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


def get_one_way_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: Union[List[str], str],
    numeric: Optional[bool] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[float]] = None,
    func: Optional[Callable] = None,
    size_cutoff: int = 0,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
):

    """
    Perform one-way analysis between two features in a DataFrame. The x_col is binned and a function is applied to the y_col grouped by values of the x_col

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_col (str):
            Name of feature on x-axis to bin
        y_col (Union[List[str], str]):
            Name of features on y-axis to measure against x-axis. Can provide up to 2 y-axis to measure against.
        numeric (Optional[bool], optional):
            Whether the x_col is numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[float]], optional):
            Optionally, provide a list of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        func (Optional[Callable], optional):
            Optionally, provide a custom function to measure y_col with
        size_cutoff (int, optional):
            Return value of NaN if x_col bin contains less than size_cutoff
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
    """
    one_way_func = (
        _get_two_one_way_analysis
        if isinstance(y_col, (list, tuple))
        else _get_one_way_analysis
    )
    return one_way_func(
        df=df,
        x_col=x_col,
        y_col=y_col,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        dp=dp,
        func=func,
        size_cutoff=size_cutoff,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
    )


def plot_one_way_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: Union[List[str], str],
    numeric: Optional[bool] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[float]] = None,
    func: Optional[Callable] = None,
    size_cutoff: int = 0,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
    x_axis_name: Optional[str] = None,
    y_axis_name: Optional[Union[List[str], str]] = None,
    plot_title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
):

    """
    Plot one-way analysis between two features in a DataFrame. The x_col is binned and a function is applied to the y_col grouped by values of the x_col

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_col (str):
            Name of feature on x-axis to bin
        y_col (Union[List[str], str]):
            Name of feature on y-axis to measure against x-axis. Can provide up to 2 y-axis to measure against.
        numeric (Optional[bool], optional):
            Whether the x_col is numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[float]], optional):
            Optionally, provide a list of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        func (Optional[Callable], optional):
            Optionally, provide a custom function to measure y_col with
        size_cutoff (int, optional):
            Return value of NaN if x_col bin contains less than size_cutoff
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
        x_axis_name (Optional[str], optional):
            Custom names to use for x_axis on plot. Defaults to None.
        y_axis_name (Optional[Union[List[str], str]], optional):
            Custom name to use for y_axis on plot. Can provide up to 2 y-axis to measure against. Defaults to None.
        plot_title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
    """

    y_col = y_col[0] if len(y_col) == 1 else y_col
    one_way_func = (
        _plot_two_one_way_analysis
        if isinstance(y_col, (list, tuple))
        else _plot_one_way_analysis
    )
    return one_way_func(
        df=df,
        x_col=x_col,
        y_col=y_col,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        dp=dp,
        func=func,
        size_cutoff=size_cutoff,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        plot_title=plot_title,
        plotsize=plotsize,
    )


def get_two_way_analysis(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    numeric: Optional[List[bool]] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[List[float]]] = None,
    func: Optional[Callable] = None,
    size_cutoff: int = 0,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
):
    """
    Perform two-way analysis between three features in a DataFrame. The x_cols are binned and a function is applied to the y_col grouped by values of the x_cols

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_cols (List[str]):
            Name of features on x-axis to bin
        y_col (Union[List[str], str]):
            Name of feature on y-axis to measure against x-axis.
        numeric (Optional[bool], optional):
            Two booleans describing whether the x_cols are numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[List[float]]], optional):
            Optionally, provide two lists of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        func (Optional[Callable], optional):
            Optionally, provide a custom function to measure y_col with
        size_cutoff (int, optional):
            Return value of NaN if x_col bin contains less than size_cutoff
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
    """

    df = df.query(filter) if filter else df
    col_1, col_2 = x_cols
    numeric_1 = (
        numeric[0]
        if numeric is not None
        else check_cont_col(df[col_1], max_card=max_card)
    )
    numeric_2 = (
        numeric[1]
        if numeric is not None
        else check_cont_col(df[col_2], max_card=max_card)
    )
    df = df[~df[col_1].isna()] if numeric_1 else df
    df = df[~df[col_2].isna()] if numeric_2 else df
    bin_1, bin_2 = ifnone(bins, get_bins(df[col_1], grid_size), get_bins(df[col_2], grid_size))
    filtered_df = df[x_cols + [y_col]].copy()
    filtered_df[col_1] = (
        pd.cut(filtered_df[col_1], bin_1, include_lowest=True)
        if numeric_1
        else filtered_df[col_1].astype("category")
    )
    filtered_df[col_2] = (
        pd.cut(filtered_df[col_2], bin_2, include_lowest=True)
        if numeric_2
        else filtered_df[col_2].astype("category")
    )
    func = ifnone(func, lambda x: conditional_mean(x, size_cutoff))

    two_way_df = (
        filtered_df.groupby(x_cols)
        .apply(func)
        .reset_index()
        .pivot(index=col_1, columns=col_2)[y_col]
    )

    if numeric_1:
        two_way_df.index = bin_intervals(
            two_way_df.index, dp, percentage, condense_last
        )

    if numeric_2:
        two_way_df.columns = bin_intervals(
            two_way_df.columns, dp, percentage, condense_last
        )
    return two_way_df


def plot_two_way_analysis(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    numeric: Optional[List[bool]] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[List[float]]] = None,
    func: Optional[Callable] = None,
    size_cutoff: int = 0,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    dep_name: Optional[str] = None,
    plot_title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    colorscale="Blues",
):

    """
    Plot two-way analysis between three features in a DataFrame. The x_cols are binned and a function is applied to the y_col grouped by values of the x_cols

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_cols (List[str]):
            Name of features on x-axis to bin
        y_col (Union[List[str], str]):
            Name of feature on y-axis to measure against x-axis.
        numeric (Optional[List[bool]], optional):
            Two booleans describing whether the x_cols are numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[List[float]]], optional):
            Optionally, provide two lists of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        func (Optional[Callable], optional):
            Optionally, provide a custom function to measure y_col with
        size_cutoff (int, optional):
            Return value of NaN if x_col bin contains less than size_cutoff
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
        feature_names (Optional[List[str]], optional):
            Custom names to use for independent variables on plot. Defaults to None.
        dep_name (Optional[str], optional):
            Custom name to use for dependent variable on plot. Defaults to None.
        plot_title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        colorscale (Union[List[str], str], optional):
            Colormap used to map scalar data to colors (for a 2D image).
            If a string is provided, it should be the name of a known color scale, and if a list is provided, it should be a list of CSS-compatible colors.
            For more information, see color_continuous_scale of https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
    """

    two_way_df = get_two_way_analysis(
        df=df,
        x_cols=x_cols,
        y_col=y_col,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        func=func,
        size_cutoff=size_cutoff,
        dp=dp,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
    )
    return plot_two_way(
        df=two_way_df,
        x_cols=x_cols,
        feature_names=feature_names,
        dep_name=dep_name,
        plot_title=plot_title,
        plotsize=plotsize,
        colorscale=colorscale,
    )


def get_two_way_frequency(
    df: pd.DataFrame,
    x_cols: List[str],
    percentage_frequency: bool = True,
    numeric: Optional[List[bool]] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[List[float]]] = None,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
):

    """
    Get frequency of two features in a DataFrame. The x_cols are binned and frequency is counted

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_cols (List[str]):
            Name of features on x-axis to bin
        percentage_frequency (bool, optional):
            Whether to return frequency as a percentage of total
        numeric (Optional[bool], optional):
            Two booleans describing whether the x_cols are numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[List[float]]], optional):
            Optionally, provide two lists of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
    """

    mod_df = df.copy()
    mod_df["dummy_count"] = 1

    output = get_two_way_analysis(
        df=mod_df,
        x_cols=x_cols,
        y_col="dummy_count",
        func=sum,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        dp=dp,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
    )
    if percentage_frequency:
        return output / output.sum()
    else:
        return output


def plot_two_way_frequency(
    df: pd.DataFrame,
    x_cols: List[str],
    percentage_frequency: bool = True,
    numeric: Optional[List[bool]] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[List[float]]] = None,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    plot_title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    colorscale="Blues",
):

    """
    Plot frequency of two features in a DataFrame. The x_cols are binned and frequency is counted

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_cols (List[str]):
            Name of features on x-axis to bin
        percentage_frequency (bool, optional):
            Whether to return frequency as a percentage of total
        numeric (Optional[bool], optional):
            Two booleans describing whether the x_cols are numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[List[float]]], optional):
            Optionally, provide two lists of values to bin x_col into, in place of quantile segmentation. Defaults to None.
        dp (int, optional):
            Decimal points to format. Defaults to 2.
        percentage (bool, optional):
            Whether to format bins as percentages. Defaults to False.
        condense_last (bool, optional):
            Whether to bin last value with a greater than. Defaults to True.
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
        feature_names (Optional[List[str]], optional):
            Custom names to use for independent variables on plot. Defaults to None.
        plot_title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        colorscale (Union[List[str], str], optional):
            Colormap used to map scalar data to colors (for a 2D image).
            If a string is provided, it should be the name of a known color scale, and if a list is provided, it should be a list of CSS-compatible colors.
            For more information, see color_continuous_scale of https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
    """
    frequency_df = get_two_way_frequency(
        df,
        x_cols,
        percentage_frequency=percentage_frequency,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        dp=dp,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
    )
    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(x_cols[0]), clean_text(x_cols[1]))
    )
    plot_title = ifnone(plot_title, f"Frequency of {feature_1} and {feature_2}")

    return plot_two_way(
        df=frequency_df,
        x_cols=x_cols,
        feature_names=feature_names,
        plot_title=plot_title,
        plotsize=plotsize,
        colorscale=colorscale,
    )


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

    def feature_correlation(self, *args, **kwargs):
        return feature_correlation(self.xs, *args, **kwargs)

    def get_one_way_analysis(self, x_col, y_col=None, *args, **kwargs):
        y_col = ifnone(y_col, self.dep_var)
        return get_one_way_analysis(self.df, x_col=x_col, y_col=y_col, *args, **kwargs)

    def plot_one_way_analysis(self, x_col, y_col=None, *args, **kwargs):
        y_col = ifnone(y_col, self.dep_var)
        return plot_one_way_analysis(self.df, x_col=x_col, y_col=y_col, *args, **kwargs)

    def get_two_way_analysis(self, x_cols, y_col=None, *args, **kwargs):
        y_col = ifnone(y_col, self.dep_var)
        return get_two_way_analysis(
            self.df, x_cols=x_cols, y_col=y_col, *args, **kwargs
        )

    def plot_two_way_analysis(self, x_cols, y_col=None, *args, **kwargs):
        y_col = ifnone(y_col, self.dep_var)
        return plot_two_way_analysis(
            self.df, x_cols=x_cols, y_col=y_col, *args, **kwargs
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


def _get_one_way_analysis(
    df,
    x_col,
    y_col,
    numeric=None,
    max_card=20,
    grid_size=20,
    bins=None,
    dp=2,
    func=None,
    size_cutoff=0,
    percentage=False,
    condense_last=True,
    filter=None,
):

    df = df.query(filter) if filter else df
    numeric = ifnone(numeric, check_cont_col(df[x_col], max_card=max_card))
    df = df[~df[x_col].isna()] if numeric else df
    bins = bins if bins else get_bins(df[x_col], grid_size)
    filtered_df = df[[x_col, y_col]].copy()

    filtered_df[x_col] = (
        pd.cut(filtered_df[x_col], bins, include_lowest=True)
        if numeric
        else filtered_df[x_col].astype("category")
    )
    func = func if func else lambda x: conditional_mean(x, size_cutoff)
    one_way_df = filtered_df.groupby(x_col).agg(
        **{y_col: (y_col, func), "size": (y_col, "count")}
    )
    if numeric:
        one_way_df.index = bin_intervals(
            one_way_df.index, dp, percentage, condense_last
        )
    return one_way_df


def _plot_one_way_analysis(
    df,
    x_col,
    y_col,
    x_axis_name=None,
    y_axis_name=None,
    plot_title=None,
    plotsize=None,
    *args,
    **kwargs,
):
    one_way_df = get_one_way_analysis(df, x_col, y_col, *args, **kwargs)
    return plot_one_way(
        df=one_way_df,
        x_col=x_col,
        y_col=y_col,
        size=one_way_df["size"],
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        plot_title=plot_title,
        plotsize=plotsize,
    )


def _get_two_one_way_analysis(
    df,
    x_col,
    y_col,
    numeric=None,
    max_card=20,
    grid_size=20,
    bins=None,
    dp=2,
    func=None,
    size_cutoff=0,
    percentage=False,
    condense_last=True,
    filter=None,
):
    if len(y_col) != 2:
        raise ValueError("Can only plot up to two columns on y-axis")

    df = df.query(filter) if filter else df
    numeric = ifnone(numeric, check_cont_col(df[x_col], max_card=max_card))
    df = df[~df[x_col].isna()] if numeric else df
    bins = bins if bins else get_bins(df[x_col], grid_size)
    filtered_df = df[[x_col] + y_col].copy()
    filtered_df[x_col] = (
        pd.cut(filtered_df[x_col], bins, include_lowest=True)
        if numeric
        else filtered_df[x_col].astype("category")
    )
    func = func if func else lambda x: conditional_mean(x, size_cutoff)
    one_way_df = filtered_df.groupby(x_col).agg(
        **{y_col[0]: (y_col[0], func), y_col[1]: (y_col[1], func)}
    )
    if numeric:
        one_way_df.index = bin_intervals(
            one_way_df.index, dp, percentage, condense_last
        )
    return one_way_df


def _plot_two_one_way_analysis(
    df,
    x_col,
    y_col,
    x_axis_name=None,
    y_axis_name=None,
    plot_title=None,
    plotsize=None,
    *args,
    **kwargs,
):
    one_way_df = _get_two_one_way_analysis(df, x_col, y_col, *args, **kwargs)
    return plot_two_one_way(
        df=one_way_df,
        x_col=x_col,
        y_col=y_col,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        plot_title=plot_title,
        plotsize=plotsize,
    )
