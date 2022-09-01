from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from FastExplain.clean.encode_categorical import EncodeCategorical
from FastExplain.clean.fill_missing import FillMissing
from FastExplain.clean.split import check_cont_col, cont_cat_split
from FastExplain.explain.bin import get_bins
from FastExplain.utils import (
    bin_intervals,
    check_list_type,
    clean_dict_text,
    clean_text,
    conditional_mean,
    doc_setter,
    fill_categorical_nan,
    ifnone,
    plot_bar,
    plot_one_way,
    plot_two_one_way,
    plot_two_way,
)


def cluster_features(xs: pd.DataFrame, threshold: float = 0.75):
    """
    Cluster features based on spearman correlation

    Args:
        xs (pd.DataFrame):
            Dataframe containing variables
        threshold (float, optional):
            Threshold to apply when forming flat clusters. Defaults to 0.75.

    """
    df = _preapre_corr_df(xs)
    corr = np.round(spearmanr(df).correlation, 4)
    corr_condensed = sch.distance.squareform(1 - corr)
    z = sch.linkage(corr_condensed, method="average")
    output = defaultdict(list)
    for i, j in zip(df.columns, sch.fcluster(z, threshold)):
        output[j].append(i)
    return list(output.values())


def feature_correlation(xs: pd.DataFrame, plotsize: List[int] = (1000, 1000)):
    """
    Plot dendogram of hierarchical clustering of spearman correlation between variables

    Args:
        xs (pd.DataFrame):
            Dataframe containing variables
        plotsize (List[int], optional):
            Custom plotsize supplied as (width, height). Defaults to (1000, 1000).
    """
    xs = _preapre_corr_df(xs)
    corr = np.round(spearmanr(xs).correlation, 4)
    fig = ff.create_dendrogram(
        1 - corr,
        orientation="left",
        labels=xs.columns,
        distfun=squareform,
        linkagefun=lambda x: sch.linkage(x, "average"),
    )
    fig.update_layout(width=plotsize[0], height=plotsize[1], plot_bgcolor="white")
    return fig


def _preapre_corr_df(xs: pd.DataFrame):
    """Prepare data for correlation analysis"""
    cont, cat = cont_cat_split(xs, max_card=1, verbose=False)
    df = FillMissing().fit_transform(xs, cont)
    df = EncodeCategorical().fit_transform(df, cat)
    keep_cols = [i for i in df.columns if len(df[i].unique()) > 1]
    keep_cols = [i for i in keep_cols if i in cont + cat]
    df = df[keep_cols]
    return df


def get_frequency(
    df: pd.DataFrame,
    x_col: str,
    numeric: Optional[bool] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[float]] = None,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
    index_mapping: Optional[dict] = None,
    display_proportion: bool = False,
):

    """
    Get frequency of feature in DataFrame. The x_col is binned and frequency is counted.

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_col (str):
            Name of feature on x-axis to bin
        numeric (Optional[bool], optional):
            Whether the x_col is numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[float]], optional):
            Optionally, provide a list of values to bin x_col into, in place of quantile segmentation. Defaults to None.
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
        index_mapping (Optional[dict], optional):
            Dictionary mapping the values to display on the x-axis
        display_proportion (bool, option):
            Whether to display as a proportion of total count
    """

    count_df = df.copy()
    count_df["count"] = 1

    filtered_df, numeric = _filter_one_way_analysis_df(
        df=count_df,
        x_col=x_col,
        y_col=["count"],
        numeric=numeric,
        bins=bins,
        filter=filter,
        max_card=max_card,
        grid_size=grid_size,
    )
    frequency_df = filtered_df.groupby(x_col).agg(**{"frequency": ("count", "count")})
    if numeric:
        frequency_df.index = bin_intervals(
            frequency_df.index, dp, percentage, condense_last
        )
    if index_mapping is not None:
        frequency_df.index = frequency_df.index.map(index_mapping)

    if display_proportion:
        frequency_df = frequency_df / frequency_df.sum()
        frequency_df.rename(columns={"frequency": "proportion"}, inplace=True)

    return frequency_df


def plot_histogram(
    df: pd.DataFrame,
    x_col: str,
    numeric: Optional[bool] = None,
    max_card: int = 20,
    grid_size: int = 20,
    bins: Optional[List[float]] = None,
    dp: int = 2,
    percentage: bool = False,
    condense_last: bool = True,
    filter: Optional[str] = None,
    index_mapping: Optional[dict] = None,
    display_proportion: bool = False,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    sort: bool = False,
    ascending: bool = True,
):
    """
    Plot frequency of feature in DataFrame. The x_col is binned and frequency is counted.

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_col (str):
            Name of feature on x-axis to bin
        numeric (Optional[bool], optional):
            Whether the x_col is numeric or categorical. If not provided, it is automatically detected based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        grid_size (int, optional):
            Number of quantiles to bin x_col into. Defaults to 20.
        bins (Optional[List[float]], optional):
            Optionally, provide a list of values to bin x_col into, in place of quantile segmentation. Defaults to None.
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
        index_mapping (Optional[dict], optional):
            Dictionary mapping the values to display on the x-axis
        display_proportion (bool, option):
            Whether to display as a proportion of total count
        xaxis_title (Optional[str], optional):
            Custom names to use for x_axis on plot. Defaults to None.
        yaxis_title (Optional[Union[List[str], str]], optional):
            Custom name to use for y_axis on plot. Can provide up to 2 y-axis to measure against. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        sort (bool, optional):
            Whether to sort values before plotting. Specify "frequency" to sort by frequency. Defaults to False
        ascending (bool, optional):
            Whether to sort ascending. Defaults to True.
    """
    xaxis_title = ifnone(xaxis_title, clean_text(x_col))
    title = ifnone(title, f"{xaxis_title} Frequency")
    frequency_df = get_frequency(
        df=df,
        x_col=x_col,
        numeric=numeric,
        max_card=max_card,
        grid_size=grid_size,
        bins=bins,
        dp=dp,
        percentage=percentage,
        condense_last=condense_last,
        filter=filter,
        index_mapping=index_mapping,
        display_proportion=display_proportion,
    )
    return plot_bar(
        df=frequency_df,
        x_col=x_col,
        y_col="proportion" if display_proportion else "frequency",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        plotsize=plotsize,
        sort=sort,
        ascending=ascending,
    )


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
    index_mapping: Optional[dict] = None,
    index_ordering: Optional[list] = None,
    legend_names: Optional[list] = None,
    histogram_include_na: bool = False,
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
        index_mapping (Optional[dict], optional):
            Dictionary mapping the values to display on the x-axis
        index_ordering (Optional[list], optional):
            List of values to order the x-axis by
        legend_names (Optional[list], optional):
            List of names to use for the legend
        histogram_include_na (bool, optional):
            Whether to include NaN values in the histogram. Defaults to False.

    """

    filtered_df, numeric = _filter_one_way_analysis_df(
        df=df,
        x_col=x_col,
        y_col=y_col,
        numeric=numeric,
        bins=bins,
        filter=filter,
        max_card=max_card,
        grid_size=grid_size,
    )

    func = func if func is not None else lambda x: conditional_mean(x, size_cutoff)

    agg_dict = _get_agg_dict(
        y_col=y_col,
        func=func,
        legend_names=legend_names,
        histogram_include_na=histogram_include_na,
    )
    one_way_df = filtered_df.groupby(x_col).agg(**agg_dict)

    if numeric:
        one_way_df.index = bin_intervals(
            one_way_df.index, dp, percentage, condense_last
        )

    if index_ordering is not None:
        one_way_df = one_way_df.loc[index_ordering]

    if index_mapping is not None:
        one_way_df.index = one_way_df.index.map(index_mapping)
    return one_way_df


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
    index_mapping: Optional[dict] = None,
    index_ordering: Optional[list] = None,
    legend_names: Optional[list] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[Union[List[str], str]] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    sort: bool = False,
    ascending: bool = True,
    display_proportion: bool = False,
    histogram_name: Optional[str] = None,
    histogram_include_na: bool = False,
    dual_axis_y_col: bool = False,
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
        index_mapping (Optional[dict], optional):
            Dictionary mapping the values to display on the x-axis. Defaults to None.
        xaxis_title (Optional[str], optional):
            Custom names to use for x_axis on plot. Defaults to None.
        yaxis_title (Optional[Union[List[str], str]], optional):
            Custom name to use for y_axis on plot. Can provide up to 2 y-axis to measure against. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        sort (bool, optional):
            Whether to sort values before plotting. Specify "frequency" to sort by frequency. Defaults to False
        ascending (bool, optional):
            Whether to sort ascending. Defaults to True.
        display_proportion (bool, optional):
            Whether to display proportion of total. Defaults to False.
        histogram_name (Optional[str], optional):
            Custom name to use for histogram. Defaults to None.
        histogram_include_na (bool, optional):
            Whether to include NaN values in the histogram. Defaults to False.
        dual_axis_y_col (bool, optional):
            Whether to plot two y-axes. Defaults to False.
    """

    y_col = y_col[0] if len(y_col) == 1 else y_col
    one_way_func = (
        _plot_two_one_way_analysis if dual_axis_y_col else _plot_one_way_analysis
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
        index_mapping=index_mapping,
        index_ordering=index_ordering,
        legend_names=legend_names,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        plotsize=plotsize,
        sort=sort,
        ascending=ascending,
        display_proportion=display_proportion,
        histogram_name=histogram_name,
        histogram_include_na=histogram_include_na,
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
    index_mapping: Optional[List[dict]] = None,
):
    """
    Perform two-way analysis between three features in a DataFrame. The x_cols are binned and a function is applied to the y_col grouped by values of the x_cols

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        x_cols (List[str]):
            Name of features on x-axis to bin
        y_col (str):
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
        index_mapping (Optional[List[dict]], optional):
            List of two dictionaries mapping the values to display on axis
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

    if bins is None:
        if numeric_1:
            bin_1 = get_bins(df[col_1], grid_size)
        if numeric_2:
            bin_2 = get_bins(df[col_2], grid_size)
    else:
        bin_1, bin_2 = bins
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
    filtered_df[col_1] = fill_categorical_nan(filtered_df[col_1])
    filtered_df[col_2] = fill_categorical_nan(filtered_df[col_2])

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

    if index_mapping is not None:
        if index_mapping[0] is not None:
            two_way_df.index = two_way_df.index.map(index_mapping[0])
        if index_mapping[1] is not None:
            two_way_df.columns = two_way_df.columns.map(index_mapping[1])

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
    index_mapping: Optional[List[dict]] = None,
    feature_names: Optional[List[str]] = None,
    dep_name: Optional[str] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    colorscale: Union[List[str], str] = "Blues",
    surface_plot: bool = True,
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
        index_mapping (Optional[List[dict]], optional):
            List of two dictionaries mapping the values to display on axis
        feature_names (Optional[List[str]], optional):
            Custom names to use for independent variables on plot. Defaults to None.
        dep_name (Optional[str], optional):
            Custom name to use for dependent variable on plot. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        colorscale (Union[List[str], str], optional):
            Colormap used to map scalar data to colors (for a 2D image).
            If a string is provided, it should be the name of a known color scale, and if a list is provided, it should be a list of CSS-compatible colors.
            For more information, see color_continuous_scale of https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
            Defaults to "Blues".
        surface_plot (bool, optional):
            Whether to plot a surface plot. Defaults to True.
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
        index_mapping=index_mapping,
    )
    return plot_two_way(
        df=two_way_df,
        x_cols=x_cols,
        feature_names=feature_names,
        dep_name=dep_name,
        title=title,
        plotsize=plotsize,
        colorscale=colorscale,
        surface_plot=surface_plot,
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
    index_mapping: Optional[List[dict]] = None,
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
        index_mapping (Optional[List[dict]], optional):
            List of two dictionaries mapping the values to display on axis
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
        index_mapping=index_mapping,
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
    index_mapping: Optional[List[dict]] = None,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    colorscale: Union[List[str], str] = "Blues",
    surface_plot: bool = True,
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
        index_mapping (Optional[List[dict]], optional):
            List of two dictionaries mapping the values to display on axis
        feature_names (Optional[List[str]], optional):
            Custom names to use for independent variables on plot. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
        colorscale (Union[List[str], str], optional):
            Colormap used to map scalar data to colors (for a 2D image).
            If a string is provided, it should be the name of a known color scale, and if a list is provided, it should be a list of CSS-compatible colors.
            For more information, see color_continuous_scale of https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
            Defaults to "Blues".
        surface_plot (bool, optional):
            Whether to plot as a surface plot. Defaults to True.
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
        index_mapping=index_mapping,
    )
    feature_1, feature_2 = ifnone(
        feature_names, (clean_text(x_cols[0]), clean_text(x_cols[1]))
    )
    title = ifnone(title, f"Frequency of {feature_1} and {feature_2}")

    return plot_two_way(
        df=frequency_df,
        x_cols=x_cols,
        feature_names=feature_names,
        title=title,
        plotsize=plotsize,
        colorscale=colorscale,
        surface_plot=surface_plot,
    )


class OneWay:
    """Connected interface for one-way and two-way analysis methods. Intended for usage with full model pipeline class. (FastExplain.models.base)"""

    def __init__(self, m, xs, df=None, dep_var=None, cat_mapping=None):
        self.m = m
        self.xs = xs
        self.df = df
        self.dep_var = dep_var
        self.cat_mapping = cat_mapping

    @doc_setter(cluster_features)
    def cluster_features(self, threshold: float = 0.75):
        return cluster_features(xs=self.xs, threshold=threshold)

    @doc_setter(feature_correlation)
    def feature_correlation(self, plotsize: List[int] = (1000, 1000)):
        return feature_correlation(xs=self.xs, plotsize=plotsize)

    @doc_setter(get_frequency)
    def get_frequency(
        self,
        x_col: str,
        numeric: Optional[bool] = None,
        max_card: int = 20,
        grid_size: int = 20,
        bins: Optional[List[float]] = None,
        dp: int = 2,
        percentage: bool = False,
        condense_last: bool = True,
        filter: Optional[str] = None,
        index_mapping: Optional[dict] = None,
        display_proportion: bool = False,
    ):

        index_mapping = self._get_index_mapping(index_mapping, x_col)
        return get_frequency(
            df=self.df,
            x_col=x_col,
            numeric=numeric,
            max_card=max_card,
            grid_size=grid_size,
            bins=bins,
            dp=dp,
            percentage=percentage,
            condense_last=condense_last,
            filter=filter,
            index_mapping=index_mapping,
            display_proportion=display_proportion,
        )

    @doc_setter(plot_histogram)
    def plot_histogram(
        self,
        x_col: str,
        numeric: Optional[bool] = None,
        max_card: int = 20,
        grid_size: int = 20,
        bins: Optional[List[float]] = None,
        dp: int = 2,
        percentage: bool = False,
        condense_last: bool = True,
        filter: Optional[str] = None,
        index_mapping: Optional[dict] = None,
        display_proportion: bool = False,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        title: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        sort: bool = False,
        ascending: bool = True,
    ):

        index_mapping = self._get_index_mapping(index_mapping, x_col)
        return plot_histogram(
            df=self.df,
            x_col=x_col,
            numeric=numeric,
            max_card=max_card,
            grid_size=grid_size,
            bins=bins,
            dp=dp,
            percentage=percentage,
            condense_last=condense_last,
            filter=filter,
            index_mapping=index_mapping,
            display_proportion=display_proportion,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            title=title,
            plotsize=plotsize,
            sort=sort,
            ascending=ascending,
        )

    @doc_setter(get_one_way_analysis)
    def get_one_way_analysis(
        self,
        x_col: str,
        y_col: Optional[Union[List[str], str]] = None,
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
        index_mapping: Optional[dict] = None,
        index_ordering: Optional[str] = None,
        legend_names: Optional[list] = None,
    ):

        y_col = ifnone(y_col, self.dep_var)
        index_mapping = self._get_index_mapping(index_mapping, x_col)

        return get_one_way_analysis(
            df=self.df,
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
            index_mapping=index_mapping,
            index_ordering=index_ordering,
            legend_names=legend_names,
        )

    @doc_setter(plot_one_way_analysis)
    def plot_one_way_analysis(
        self,
        x_col: str,
        y_col: Optional[Union[List[str], str]] = None,
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
        index_mapping: Optional[dict] = None,
        index_ordering: Optional[list] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[Union[List[str], str]] = None,
        title: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        sort: bool = False,
        ascending: bool = True,
        display_proportion: bool = False,
        histogram_name: Optional[str] = None,
        legend_names: Optional[list] = None,
        histogram_include_na: bool = False,
        dual_axis_y_col: bool = False,
    ):

        y_col = ifnone(y_col, self.dep_var)
        index_mapping = self._get_index_mapping(index_mapping, x_col)
        return plot_one_way_analysis(
            df=self.df,
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
            index_mapping=index_mapping,
            index_ordering=index_ordering,
            legend_names=legend_names,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            title=title,
            plotsize=plotsize,
            sort=sort,
            ascending=ascending,
            display_proportion=display_proportion,
            histogram_name=histogram_name,
            histogram_include_na=histogram_include_na,
            dual_axis_y_col=dual_axis_y_col,
        )

    @doc_setter(get_two_way_analysis)
    def get_two_way_analysis(
        self,
        x_cols: List[str],
        y_col: Optional[str] = None,
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
        index_mapping: Optional[List[dict]] = None,
    ):

        y_col = ifnone(y_col, self.dep_var)
        index_mapping = self._get_two_index_mapping(index_mapping, x_cols)
        return get_two_way_analysis(
            df=self.df,
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
            index_mapping=index_mapping,
        )

    @doc_setter(plot_two_way_analysis)
    def plot_two_way_analysis(
        self,
        x_cols: List[str],
        y_col: Optional[str] = None,
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
        index_mapping: Optional[List[dict]] = None,
        feature_names: Optional[List[str]] = None,
        dep_name: Optional[str] = None,
        title: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        colorscale: Union[List[str], str] = "Blues",
        surface_plot: bool = True,
    ):

        y_col = ifnone(y_col, self.dep_var)
        index_mapping = self._get_two_index_mapping(index_mapping, x_cols)
        return plot_two_way_analysis(
            df=self.df,
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
            index_mapping=index_mapping,
            feature_names=feature_names,
            dep_name=dep_name,
            title=title,
            plotsize=plotsize,
            colorscale=colorscale,
            surface_plot=surface_plot,
        )

    @doc_setter(get_two_way_frequency)
    def get_two_way_frequency(
        self,
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
        index_mapping: Optional[List[dict]] = None,
    ):

        index_mapping = self._get_two_index_mapping(index_mapping, x_cols)
        return get_two_way_frequency(
            df=self.df,
            x_cols=x_cols,
            percentage_frequency=percentage_frequency,
            numeric=numeric,
            max_card=max_card,
            grid_size=grid_size,
            bins=bins,
            dp=dp,
            percentage=percentage,
            condense_last=condense_last,
            filter=filter,
            index_mapping=index_mapping,
        )

    @doc_setter(plot_two_way_frequency)
    def plot_two_way_frequency(
        self,
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
        index_mapping: Optional[List[dict]] = None,
        feature_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        plotsize: Optional[List[int]] = None,
        colorscale: Union[List[str], str] = "Blues",
        surface_plot: bool = True,
    ):

        index_mapping = self._get_two_index_mapping(index_mapping, x_cols)
        return plot_two_way_frequency(
            df=self.df,
            x_cols=x_cols,
            percentage_frequency=percentage_frequency,
            numeric=numeric,
            max_card=max_card,
            grid_size=grid_size,
            bins=bins,
            dp=dp,
            percentage=percentage,
            condense_last=condense_last,
            filter=filter,
            index_mapping=index_mapping,
            feature_names=feature_names,
            title=title,
            plotsize=plotsize,
            colorscale=colorscale,
            surface_plot=surface_plot,
        )

    def _get_index_mapping(self, index_mapping, x_col):
        return ifnone(
            index_mapping,
            clean_dict_text(self.cat_mapping[x_col])
            if x_col in self.cat_mapping
            else None,
        )

    def _get_two_index_mapping(self, index_mapping, x_cols):
        return ifnone(
            index_mapping,
            [
                clean_dict_text(self.cat_mapping[x_cols[0]])
                if x_cols[0] in self.cat_mapping
                else None,
                clean_dict_text(self.cat_mapping[x_cols[1]])
                if x_cols[1] in self.cat_mapping
                else None,
            ],
        )


def _filter_one_way_analysis_df(
    df, x_col, y_col, numeric, bins, filter, max_card, grid_size
):

    y_col = y_col if isinstance(y_col, (list, tuple)) else [y_col]

    df = df.query(filter) if filter else df

    numeric = ifnone(numeric, check_cont_col(df[x_col], max_card=max_card))

    if numeric:
        bins = bins if bins else get_bins(df[x_col], grid_size)

    filtered_df = df[[x_col] + y_col].copy()
    filtered_df[x_col] = (
        pd.cut(filtered_df[x_col], bins, include_lowest=True)
        if numeric
        else filtered_df[x_col].astype("category")
    )

    filtered_df[x_col] = fill_categorical_nan(filtered_df[x_col])
    return filtered_df, numeric


def _get_agg_dict(y_col, func, legend_names, histogram_include_na):

    histogram_func = len if histogram_include_na else "count"

    if check_list_type(func) and check_list_type(y_col):
        raise NotImplementedError("Can only have one of function or y_col as lists")

    if check_list_type(func):
        legend_names = ifnone(
            legend_names, [f"{y_col} Metric {i}" for i in range(len(func))]
        )
        agg_dict = {k: (y_col, v) for k, v in zip(legend_names, func)}
        agg_dict["size"] = (y_col, histogram_func)

    elif check_list_type(y_col):
        legend_names = ifnone(legend_names, y_col)
        agg_dict = {k: (v, func) for k, v in zip(legend_names, y_col)}
        agg_dict["size"] = (y_col[0], histogram_func)

    else:
        legend_names = ifnone(legend_names, y_col)
        agg_dict = {legend_names: (y_col, func)}
        agg_dict["size"] = (y_col, histogram_func)

    return agg_dict


def _plot_one_way_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: Union[list, str],
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    sort: bool = False,
    ascending: bool = True,
    display_proportion: bool = False,
    histogram_name: Optional[str] = None,
    *args,
    **kwargs,
):

    """Base function to plot one way analysis"""

    one_way_df = get_one_way_analysis(df=df, x_col=x_col, y_col=y_col, *args, **kwargs)
    return plot_one_way(
        df=one_way_df,
        x_col=x_col,
        y_col=y_col,
        size=one_way_df["size"],
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        plotsize=plotsize,
        sort=sort,
        ascending=ascending,
        display_proportion=display_proportion,
        histogram_name=histogram_name,
    )


def _plot_two_one_way_analysis(
    df: pd.DataFrame,
    x_col: str,
    y_col: List[str],
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[List[str]] = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
    sort=False,
    ascending=True,
    display_proportion: bool = False,
    histogram_name: Optional[str] = None,
    histogram_include_na: bool = False,
    *args,
    **kwargs,
):

    """Base function to plot one way analysis for two features"""
    one_way_df = get_one_way_analysis(df=df, x_col=x_col, y_col=y_col, *args, **kwargs)
    return plot_two_one_way(
        df=one_way_df,
        x_col=x_col,
        y_col=y_col,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        plotsize=plotsize,
    )
