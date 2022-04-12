from random import sample
import warnings
from modelflow.utils import check_unequal_list


def sensitivity_test(
    df,
    m=None,
    m_cols: list = [],
    trials: int = 50,
    replace_features: list = [],
    replacement_conditions: list = [],
    replacement_values: list = [],
    percent_replaces=None,
    metric_agg_func=sum,
    pred_func=None,
    replace_func=None,
):

    if pred_func is None:

        def pred_func(df):
            return predict_column(df, m, m_cols)

    if replace_func is None:

        def replace_func(df):
            _replace_features(
                df,
                replace_features,
                replacement_conditions,
                replacement_values,
                percent_replaces,
            )

    trial_results = []

    for _ in range(trials):
        sensitivity_df = df.copy()

        # Predict Old
        old_pred = pred_func(sensitivity_df)
        old_metric = metric_agg_func(old_pred)

        # Replace Values
        replace_func(sensitivity_df)

        # Predict New
        new_pred = pred_func(sensitivity_df)
        new_metric = metric_agg_func(new_pred)
        trial_results.append(new_metric / old_metric)

    return trial_results


def _replace_features(
    df,
    replace_features,
    replacement_conditions,
    replacement_values,
    percent_replaces=None,
):

    percent_replaces = (
        percent_replaces if percent_replaces else [0.2 for _ in replace_features]
    )

    if check_unequal_list(
        [
            replace_features,
            replacement_conditions,
            replacement_values,
            percent_replaces,
        ]
    ):
        raise ValueError("Replacement inputs are not equal")

    for (
        replace_feature,
        replacement_condition,
        replacement_value,
        percent_replace,
    ) in zip(
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
            percent_replace,
        )
    return


def _replace_feature(
    df,
    replace_feature,
    replacement_condition,
    replacement_value,
    percent_replace=0.2,
):
    non_index = df.query(replacement_condition).index
    sample_index = sample(list(non_index), round(len(non_index) * percent_replace))
    df.loc[sample_index, replace_feature] = replacement_value
    return


def predict_column(df, m, m_cols):
    return m.predict(df[m_cols])


class Sensitivity:
    def __init__(self, m, xs, m_cols=None):
        self.m = m
        self.xs = xs
        self.m_cols = m_cols
        if self.m_cols is None:
            warnings.warn("Sensitivity analysis does not work without model columns")

    def sensitivity_test(self, *args, **kwargs):
        return sensitivity_test(self.xs, self.m, self.m_cols, *args, **kwargs)
