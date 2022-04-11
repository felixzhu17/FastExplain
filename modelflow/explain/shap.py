import shap
import pandas as pd


def get_shap_values(m, xs):
    explainer = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(xs)
    shap_values_df = pd.DataFrame(
        shap_values, columns=["shap_" + i for i in xs.columns]
    )
    shap_values_df.index = xs.index
    return shap_values_df


class ShapExplain:
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    def get_shap_values(self, *args, **kwargs):
        return get_shap_values(self.m, self.xs, *args, **kwargs)
