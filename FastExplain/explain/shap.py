import pandas as pd
import shap

from FastExplain.utils import ifnone, query_df_index, sample_index

SHAP_MAX_SAMPLES = 10_000


class ShapExplain:
    def __init__(self, m, xs, shap_max_samples=SHAP_MAX_SAMPLES):
        self.m = m
        self.xs = xs
        self.shap_max_samples = shap_max_samples
        self.sample_seed = 0
        shap.initjs()

    @property
    def sample_index(self):
        if len(self.xs) > self.shap_max_samples:
            return sample_index(
                self.xs, n=self.shap_max_samples, random_state=self.sample_seed
            )
        else:
            return list(self.xs.index)

    def shap_force_plot(self, filter=None, index=[], shap_max_samples=None):
        self.set_shap_values()
        self.shap_max_samples = ifnone(shap_max_samples, self.shap_max_samples)
        index = self.get_shap_index(filter, index)
        return shap.plots.force(self.shap_values[index])

    def shap_summary_plot(self, filter=None, index=[], shap_max_samples=None):
        self.set_shap_values()
        self.shap_max_samples = ifnone(shap_max_samples, self.shap_max_samples)
        index = self.get_shap_index(filter, index)
        return shap.plots.beeswarm(self.shap_values[index])

    def shap_dependence_plot(self, col, filter=None, index=[], shap_max_samples=None):
        self.set_shap_values()
        self.shap_max_samples = ifnone(shap_max_samples, self.shap_max_samples)
        index = self.get_shap_index(filter, index)
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

    def _get_shap_values_df(self):
        shap_values_df = pd.DataFrame(
            self.shap_values.values, columns=["shap_" + i for i in self.xs.columns]
        )
        shap_values_df.index = self.xs.index
        return shap_values_df

    def get_shap_index(self, filter, index):
        if filter:
            return query_df_index(self.xs, filter)[: self.shap_max_samples]
        elif len(index) > 0:
            return index
        else:
            return self.sample_index
