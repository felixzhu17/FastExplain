import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans

from FastExplain.clean.split import cont_cat_split
from FastExplain.utils import (
    ifnone,
    is_rf,
    most_common_values,
    query_df_index,
    sample_index,
)

SHAP_MAX_SAMPLES = 1_000_000


class Shap:
    """
    Class used to prepare and plot SHAP values

    Args:
        m (type):
            Fitted model
        xs (pd.DataFrame):
            Dataframe used by model to predict.
        shap_max_samples (int, optional):
            Maximum samples for plotting SHAP. Defaults to 10000.
        sample_seed (int, optional):
            Seed used to sample index. Defaults to 0.

    Attributes:
        m:
            Fitted model
        xs:
            Dataframe used by model to predict. Index is reset.
        shap_max_samples:
            Maximum samples for plotting SHAP
        sample_seed:
            Seed used to sample index
        shap_values:
            Calculated SHAP values. Available after get_shap_values or any plotting function is called to save memory.
        shap_value_df:
            SHAP values merged with xs. Available after get_shap_values or any plotting function is called to save memory.
    """

    def __init__(
        self,
        m: type,
        xs: pd.DataFrame,
        dep_var=None,
        cat_mapping=None,
        y=None,
        shap_max_samples: int = SHAP_MAX_SAMPLES,
    ):
        self.m = m
        if is_rf(m):
            warnings.warn(
                "SHAP values is very slow for RandomForest. Use XGBoost instead for faster performance"
            )
        self.xs = xs
        self.shap_max_samples = shap_max_samples
        self.sample_seed = 0
        self.cat_mapping = cat_mapping
        self.y = y
        self.dep_var = dep_var
        self.shap_values_df = None
        shap.initjs()

    @property
    def sample_index(self):
        """Sample index used for plotting"""
        if len(self.xs) > self.shap_max_samples:
            return sample_index(
                self.xs, n=self.shap_max_samples, random_state=self.sample_seed
            )
        else:
            return list(self.xs.index)

    def shap_force_plot(
        self,
        filter: Optional[str] = None,
        index: Optional[List[Any]] = None,
        shap_max_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Visualize the given SHAP values with an additive force layout.

        Args:
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
            index (Optional[List[Any]], optional):
                Provide index for shap_values. Defaults to None.
            shap_max_samples (Optional[int], optional):
                Maximum number of samples for plotting. If provided, sets the new default for other plots. If None, uses default from class. Defaults to None.
            *args, **kwargs:
                Additional arguments for SHAP Force Plots. See https://shap.readthedocs.io/en/latest/generated/shap.plots.force.html?highlight=force
        """

        _ = self.get_shap_values(shap_max_samples)
        index = self.get_shap_index(filter, index)
        return shap.plots.force(self.shap_values[index], *args, **kwargs)

    def shap_summary_plot(
        self,
        filter: Optional[str] = None,
        index: Optional[List[Any]] = None,
        shap_max_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Create a SHAP beeswarm plot, colored by feature values when they are provided.

        Args:
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
            index (Optional[List[Any]], optional):
                Provide index for shap_values. Defaults to None.
            shap_max_samples (Optional[int], optional):
                Maximum number of samples for plotting. If provided, sets the new default for other plots. If None, uses default from class. Defaults to None.
            *args, **kwargs:
                Additional arguments for SHAP Beeswarm Plots. See https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html?highlight=BEESWARM
        """

        _ = self.get_shap_values(shap_max_samples)
        index = self.get_shap_index(filter, index)
        return shap.plots.beeswarm(self.shap_values[index], *args, **kwargs)

    def shap_dependence_plot(
        self,
        col,
        filter: Optional[str] = None,
        index: Optional[List[Any]] = None,
        shap_max_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Create a SHAP dependence scatter plot, colored by an interaction feature.

        Args:
            col (str):
                Name of predictor feature to use for SHAP
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
            index (Optional[List[Any]], optional):
                Provide index for shap_values. Defaults to None.
            shap_max_samples (Optional[int], optional):
                Maximum number of samples for plotting. If provided, sets the new default for other plots. If None, uses default from class. Defaults to None.
            *args, **kwargs:
                Additional arguments for SHAP Scatter Plots. See https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html#Simple-dependence-scatter-plot
        """

        _ = self.get_shap_values(shap_max_samples)
        index = self.get_shap_index(filter, index)
        return shap.plots.scatter(
            self.shap_values[index, col], color=self.shap_values[index], *args, **kwargs
        )

    def shap_importance_plot(
        self,
        shap_max_samples: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Create a bar plot of a set of SHAP values.

        Args:
            *args, **kwargs:
                Additional arguments for SHAP Bar Plots. See https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html#
        """
        _ = self.get_shap_values(shap_max_samples)
        return shap.plots.bar(self.shap_values, *args, **kwargs)

    def get_shap_values(self, shap_max_samples: Optional[int] = None):
        """If SHAP values already calculated, pass, else calculate SHAP values. Return SHAP values"""
        self.shap_max_samples = ifnone(shap_max_samples, self.shap_max_samples)
        if self._explainer_exists() and self._shap_values_exists():
            pass
        else:
            self._set_shap_values()
        return self.shap_values_df

    def cluster_shap_values(self, n, shap_max_samples: Optional[int] = None):
        _ = self.get_shap_values(shap_max_samples)
        cluster_df = self.shap_values_df.copy()
        shap_cols = [i for i in cluster_df.columns if i.startswith("shap")]
        kmeans = KMeans(n_clusters=n).fit(cluster_df[shap_cols])
        cluster_df["cluster"] = kmeans.labels_
        return cluster_df

    def shap_cluster_summary(
        self,
        n,
        cat_top_cutoff=3,
        cont_func="median",
        shap_max_samples: Optional[int] = None,
    ):
        cluster_df = self.cluster_shap_values(n, shap_max_samples)
        summary_df = cluster_df[
            [i for i in cluster_df.columns if not i.startswith("shap_")]
        ]
        cont, cat = cont_cat_split(summary_df, dep_var="cluster")
        summary_df["count"] = 1
        agg_funcs = {
            **{i: cont_func for i in cont},
            **{i: lambda x: most_common_values(x, cat_top_cutoff) for i in cat},
            "count": "size",
        }
        return summary_df.groupby("cluster").agg(agg_funcs)

    def _set_shap_values(self):
        """Calculate SHAP values"""
        self.explainer = shap.Explainer(self.m)
        if len(self.explainer.expected_value) == 1:
            self.explainer.expected_value = self.explainer.expected_value.mean()
        self.shap_values = self.explainer(self.xs)
        if len(self.shap_values.shape) == 3:
            self.shap_values = self.shap_values[:, :, 1]
        self.shap_values_df = self._get_shap_values_df()

    def _explainer_exists(self):
        """Check if explainer created"""
        return hasattr(self, "explainer")

    def _shap_values_exists(self):
        """Check if SHAP calculated"""
        return hasattr(self, "shap_values")

    def _get_shap_values_df(self):
        """Merge SHAP values with xs"""
        shap_values_df = pd.DataFrame(
            self.shap_values.values, columns=["shap_" + i for i in self.xs.columns]
        )
        shap_values_df.index = self.xs.index
        shap_df = self.xs.merge(shap_values_df, left_index=True, right_index=True)
        if self.cat_mapping is not None:
            for k, v in self.cat_mapping.items():
                if k in shap_df:
                    shap_df[k] = shap_df[k].map(v)
        if self.y is not None:
            shap_df = shap_df.join(self.y)
        return shap_df

    def get_shap_index(self, filter, index):
        """Sample index based on filter or index or default"""
        if filter:
            return query_df_index(self.xs, filter)[: self.shap_max_samples]
        elif index:
            return index
        else:
            return self.sample_index
