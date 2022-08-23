from random import sample
from typing import Any, Callable, List, Optional, Union

import pandas as pd

from FastExplain.utils import check_unequal_list, doc_setter, fill_list, ifnone


def sensitivity_test(
    df: pd.DataFrame,
    m: type,
    replace_features: List[str],
    replacement_values: List[Any],
    trials: int = 50,
    replacement_conditions: Optional[List[str]] = None,
    relative_diff: bool = True,
    percent_replace: Union[List[float], float] = 1,
    metric_agg_func: Callable = sum,
    replace_func: Optional[Callable] = None,
):
    """
    Analyse what happens to a metric of the dependent variable when modifications are made to the data

    Args:
        df (pd.DataFrame):
            Dataframe used by model to predict.
        m (type):
            Trained model that uses features as a predictor.
        replace_features (List[str]):
            List of features to replace for sensitvity test
        replacement_values (List[Any]):
            List of values to replace features with for sensitivity test. Must correspond with replace_features.
        trials (int, optional):
            Number of trials to run for sensitivity test. Defaults to 50.
        replacement_conditions (Optional[List[str]], optional):
            List of conditions for observations to replace features. If supplied, must correspond with replace_features (for features without conditions, use None). If not supplied, all observations may be replaced.
            Condition must be in the following format:
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
        relative_diff (bool, optional):
            Whether to calculate relative difference of metric or absolute difference of metric. Defaults to True.
        percent_replace (Union[List[float], float], optional):
            Percentage of observations meeting the replacement conditions to replace. Defaults to 0.2.
        metric_agg_func (Callable, optional):
            Function for aggregating list of predictions. Defaults to sum.
        replace_func (Optional[Callable], optional):
            Optional function for replacing features. Function must contain df as an argument. Defaults to None.
    """

    if replace_func is None:

        def replace_func(df):
            _replace_features(
                df,
                replace_features,
                replacement_conditions,
                replacement_values,
                percent_replace,
            )

    trial_results = []

    for _ in range(trials):
        sensitivity_df = df.copy()

        # Predict Old
        old_pred = m.predict(sensitivity_df)
        old_metric = metric_agg_func(old_pred)

        # Replace Values
        replace_func(sensitivity_df)

        # Predict New
        new_pred = m.predict(sensitivity_df)
        new_metric = metric_agg_func(new_pred)
        trial_results.append(
            new_metric / old_metric if relative_diff else new_metric - old_metric
        )

    return trial_results


def _replace_features(
    df,
    replace_features,
    replacement_conditions,
    replacement_values,
    percent_replace,
):
    """Default function for replacing list of features"""

    percent_replaces = (
        percent_replace
        if isinstance(percent_replace, (list, tuple))
        else fill_list([], len(replace_features), percent_replace)
    )

    replacement_conditions = ifnone(
        replacement_conditions, fill_list([], len(replace_features), None)
    )

    if check_unequal_list(
        [replace_features, replacement_conditions, replacement_values, percent_replaces]
    ):
        raise ValueError("Replacement inputs are not equal")

    for (replace_feature, replacement_condition, replacement_value, percent,) in zip(
        replace_features,
        replacement_conditions,
        replacement_values,
        percent_replaces,
    ):
        _replace_feature(
            df,
            replace_feature,
            replacement_condition,
            replacement_value,
            percent,
        )
    return


def _replace_feature(
    df,
    replace_feature,
    replacement_condition,
    replacement_value,
    percent,
):
    """Default function for replacing feature"""
    non_index = (
        df.query(replacement_condition).index if replacement_condition else df.index
    )
    sample_index = sample(list(non_index), round(len(non_index) * percent))
    df.loc[sample_index, replace_feature] = replacement_value
    return


class Sensitivity:
    """Connected interface for Sensitivity methods. Intended for usage with full model pipeline class. (FastExplain.models.base)"""

    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    @doc_setter(sensitivity_test)
    def sensitivity_test(
        self,
        replace_features: List[str],
        replacement_values: List[Any],
        trials: int = 50,
        replacement_conditions: Optional[List[str]] = None,
        relative_diff: bool = True,
        percent_replace: Union[List[float], float] = 0.2,
        metric_agg_func: Callable = sum,
        replace_func: Optional[Callable] = None,
    ):

        return sensitivity_test(
            df=self.xs,
            m=self.m,
            replace_features=replace_features,
            replacement_values=replacement_values,
            trials=trials,
            replacement_conditions=replacement_conditions,
            relative_diff=relative_diff,
            percent_replace=percent_replace,
            metric_agg_func=metric_agg_func,
            replace_func=replace_func,
        )
