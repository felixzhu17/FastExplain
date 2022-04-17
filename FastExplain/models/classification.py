from typing import List, Optional

import pandas as pd

from FastExplain.clean import prepare_data
from FastExplain.explain import Explain
from FastExplain.metrics import (
    auc,
    confusion_matrix,
    get_error,
    m_cross_entropy,
    plot_one_way_error,
    plot_two_way_error,
)
from FastExplain.models.algorithms import (
    ebm_class,
    get_model_parameters,
    rf_class,
    xgb_class,
)
from FastExplain.utils import ifnone

CLASS_MODELS = {
    "rf": rf_class,
    "xgb": xgb_class,
    "ebm": ebm_class,
}


class Classification(
    Explain,
):
    def __init__(
        self,
        df: pd.DataFrame,
        dep_var: str,
        cat_names: Optional[List[str]] = None,
        cont_names: Optional[List[str]] = None,
        model: str = "rf",
        perc_train: int = 0.8,
        seed: int = 0,
        splits: Optional[List[List]] = None,
        cat_strategy: str = "ordinal",
        fill_strategy: str = "median",
        fill_const: int = 0,
        na_dummy: bool = True,
        cont_transformations: List[type] = [],
        reduce_memory: bool = True,
        *args,
        **kwargs,
    ):

        self.data = prepare_data(
            df=df,
            cat_names=cat_names,
            cont_names=cont_names,
            dep_var=dep_var,
            perc_train=perc_train,
            seed=seed,
            splits=splits,
            cat_strategy=cat_strategy,
            fill_strategy=fill_strategy,
            fill_const=fill_const,
            na_dummy=na_dummy,
            cont_transformations=cont_transformations,
            reduce_memory=reduce_memory,
            return_class=True,
        )

        if model in CLASS_MODELS:

            self.m = CLASS_MODELS[model](
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Model can only be one of {', '. join(CLASS_MODELS)}")

        self.error = self._get_error()
        self.raw_error = self._get_raw_error()
        self.params = get_model_parameters(self.m)

        Explain.__init__(
            self,
            self.m,
            self.data.xs,
            self.data.df,
            self.data.dep_var,
            self.data.train_xs.columns,
        )

    def plot_auc(self, val: bool = True, *args, **kwargs):
        if val:
            return auc(
                self.m, self.data.val_xs, self.data.val_y, plot=True, *args, **kwargs
            )
        else:
            return auc(
                self.m,
                self.data.xs,
                self.data.y,
                plot=True,
                *args,
                **kwargs,
            )

    def confusion_matrix(self, val: bool = True, *args, **kwargs):
        if val:
            return confusion_matrix(
                self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
            )
        else:
            return confusion_matrix(self.m, self.data.xs, self.data.y, *args, **kwargs)

    def plot_one_way_error(self, col: Optional[str] = None, *args, **kwargs):
        col = ifnone(col, self.data.dep_var)
        return plot_one_way_error(
            self.data.df,
            self.raw_error["cross_entropy"]["model"]["overall"],
            col,
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols: List[str], *args, **kwargs):
        return plot_two_way_error(
            self.data.df,
            self.raw_error["cross_entropy"]["model"]["overall"],
            cols,
            *args,
            **kwargs,
        )

    def _get_error(self):
        auc_score = {
            "model": get_error(
                auc,
                self.m,
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                self.data.xs,
                self.data.y,
                False,
            ),
        }

        cross_entropy = {
            "model": get_error(
                m_cross_entropy,
                self.m,
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                self.data.xs,
                self.data.y,
                True,
            )
        }

        return {"auc": auc_score, "cross_entropy": cross_entropy}

    def _get_raw_error(self):
        cross_entropy_prob = {
            "model": get_error(
                m_cross_entropy,
                self.m,
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                self.data.xs,
                self.data.y,
                False,
            )
        }
        return {"cross_entropy": cross_entropy_prob}
