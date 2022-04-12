import shap
import pandas as pd
from modelflow.utils import query_df_index, sample_index

SHAP_MAX_SAMPLES = 10_000


class ShapExplain:
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs
        self.shap_max_samples = SHAP_MAX_SAMPLES
        self.sample_index = self._set_sample_index()
        shap.initjs()

    def shap_force_plot(self, query=None, index=[]):
        self.set_shap_values()
        index = self.get_shap_index(query, index)
        return shap.plots.force(self.shap_values[index])

    def shap_summary_plot(self, query=None, index=[]):
        self.set_shap_values()
        index = self.get_shap_index(query, index)
        return shap.plots.beeswarm(self.shap_values[index])

    def shap_dependence_plot(self, col, query=None, index=[]):
        self.set_shap_values()
        index = self.get_shap_index(query, index)
        return shap.plots.scatter(
            self.shap_values[index, col], color=self.shap_values[index]
        )

    def shap_importance_plot(self):
        self.set_shap_values()
        return shap.plots.bar(self.shap_values)

    def set_shap_values(self):
        if self._explainer_exists() and self._shap_values_exists():
            pass
        else:
            self._set_shap_values()

    def _set_shap_values(self):
        self.explainer = shap.Explainer(self.m)
        if len(self.explainer.expected_value) == 1:
            self.explainer.expected_value = self.explainer.expected_value.mean()
        self.shap_values = self.explainer(self.xs)
        if len(self.shap_values.shape) == 3:
            self.shap_values = self.shap_values[:, :, 1]
        self.shap_value_df = self._get_shap_values_df()
        self.xs = self.xs.reset_index(drop=True)

    def _explainer_exists(self):
        return hasattr(self, "explainer")

    def _shap_values_exists(self):
        return hasattr(self, "shap_values")

    def _set_sample_index(self, seed=0):
        if len(self.xs) > self.shap_max_samples:
            return sample_index(self.xs, n=self.shap_max_samples, random_state=seed)
        else:
            return list(self.xs.index)

    def _get_shap_values_df(self):
        shap_values_df = pd.DataFrame(
            self.shap_values.values, columns=["shap_" + i for i in self.xs.columns]
        )
        shap_values_df.index = self.xs.index
        return shap_values_df

    def get_shap_index(self, query, index):
        if query:
            return query_df_index(self.xs, query)
        elif len(index) > 0:
            return index
        else:
            return self.sample_index
