from typing import Callable, List, Optional

import pandas as pd

from FastExplain.explain.one_way import get_one_way_analysis, get_two_way_analysis
from FastExplain.utils import plot_one_way, plot_two_way


def get_benchmark_error(func, benchmark, train_y, val_y, y, mean):
    """Function to get all benchmark error"""
    return {
        "train": func(benchmark, train_y, mean),
        "val": func(benchmark, val_y, mean),
        "overall": func(benchmark, y, mean),
    }


def get_error(func, m, train_xs, train_y, val_xs, val_y, xs, y, mean):
    """Function to get all model error"""
    return {
        "train": func(m, train_xs, train_y, mean),
        "val": func(m, val_xs, val_y, mean),
        "overall": func(m, xs, y, mean),
    }


def get_one_way_error(
    df: pd.DataFrame,
    error: List[float],
    x_col: str,
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
    Perform one-way analysis between model error and an error in a DataFrame. The x_col is binned and a function is applied to the error grouped by values of the x_col

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        error (List[float]):
            Array of errors from fitted model
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
    error_df = df.copy()
    error_df["error"] = error
    return get_one_way_analysis(
        df=error_df,
        x_col=x_col,
        y_col="error",
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


def plot_one_way_error(
    df: pd.DataFrame,
    error: List[float],
    x_col: str,
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
    xaxis_title: str = None,
    yaxis_title: str = None,
    title: Optional[str] = None,
    plotsize: Optional[List[int]] = None,
):

    """
    Plot one-way analysis between model error and an error in a DataFrame. The x_col is binned and a function is applied to the error grouped by values of the x_col

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        error (List[float]):
            Array of errors from fitted model
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
        xaxis_title (Optional[str], optional):
            Custom names to use for x_axis on plot. Defaults to None.
        yaxis_title (Optional[Union[List[str], str]], optional):
            Custom name to use for y_axis on plot. Can provide up to 2 y-axis to measure against. Defaults to None.
        title (Optional[str], optional):
            Custom name to use for title of plot. Defaults to None.
        plotsize (Optional[List[int]], optional):
            Custom plotsize supplied as (width, height). Defaults to None.
    """
    one_way_error_df = get_one_way_error(
        df=df,
        error=error,
        x_col=x_col,
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
    return plot_one_way(
        df=one_way_error_df,
        x_col=x_col,
        y_col="error",
        size=one_way_error_df["size"],
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        plotsize=plotsize,
    )


def get_two_way_error(
    df: pd.DataFrame,
    error: List[float],
    x_cols: List[str],
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
    Perform two-way analysis between error and two features in a DataFrame. The x_cols are binned and a function is applied to the error grouped by values of the x_cols

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        error (List[float]):
            Array of errors from fitted model
        x_cols (List[str]):
            Name of features on x-axis to bin
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

    error_df = df.copy()
    error_df["error"] = error
    return get_two_way_analysis(
        df=error_df,
        x_cols=x_cols,
        y_col="error",
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


def plot_two_way_error(
    df: pd.DataFrame,
    error: List[float],
    x_cols: List[str],
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
    feature_names=None,
    dep_name=None,
    title=None,
    plotsize=None,
    colorscale="Blues",
    surface_plot: bool = True,
):

    """
    Perform two-way analysis between error and two features in a DataFrame. The x_cols are binned and a function is applied to the error grouped by values of the x_cols

    Args:
        df (pd.DataFrame):
            Dataframe containing features to compare
        error (List[float]):
            Array of errors from fitted model
        x_cols (List[str]):
            Name of features on x-axis to bin
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
    """
    output = get_two_way_error(
        df,
        error,
        x_cols,
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
        df=output,
        x_cols=x_cols,
        feature_names=feature_names,
        dep_name=dep_name,
        title=title,
        plotsize=plotsize,
        colorscale=colorscale,
        surface_plot=surface_plot,
    )
