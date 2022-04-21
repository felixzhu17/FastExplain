from typing import Callable, List, Optional, Union

import pandas as pd

from FastExplain.metrics import (
    auc,
    confusion_matrix,
    cross_entropy,
    get_benchmark_error,
    get_error,
    m_cross_entropy,
    plot_one_way_error,
    plot_two_way_error,
)
from FastExplain.models.algorithms import ebm_class, rf_class, xgb_class
from FastExplain.models.base import Model
from FastExplain.utils import ifnone

CLASS_MODELS = {
    "rf": rf_class,
    "xgb": xgb_class,
    "ebm": ebm_class,
}


class Classification(Model):
    def __init__(
        self,
        df: pd.DataFrame,
        dep_var: str,
        cat_names: Optional[List[str]] = None,
        cont_names: Optional[List[str]] = None,
        model: Union[str, type, Callable] = "rf",
        perc_train: int = 0.8,
        seed: int = 0,
        splits: Optional[List[List]] = None,
        cat_strategy: str = "ordinal",
        fill_strategy: str = "median",
        fill_const: int = 0,
        na_dummy: bool = True,
        cont_transformations: List[type] = [],
        reduce_memory: bool = True,
        hypertune=False,
        hypertune_max_evals=100,
        hypertune_params=None,
        hypertune_loss_metric=None,
        *model_args,
        **model_kwargs,
    ):
        self.classification = True
        Model.__init__(
            self,
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            model=model,
            default_models=CLASS_MODELS,
            perc_train=perc_train,
            seed=seed,
            splits=splits,
            cat_strategy=cat_strategy,
            fill_strategy=fill_strategy,
            fill_const=fill_const,
            na_dummy=na_dummy,
            cont_transformations=cont_transformations,
            reduce_memory=reduce_memory,
            hypertune=hypertune,
            hypertune_max_evals=hypertune_max_evals,
            hypertune_params=hypertune_params,
            hypertune_loss_metric=hypertune_loss_metric,
            *model_args,
            **model_kwargs,
        )

    def plot_auc(self, val: bool = True, *args, **kwargs):
        if val:
            return auc(
                m=self.m,
                xs=self.data.val_xs,
                y=self.data.val_y,
                plot=True,
                *args,
                **kwargs,
            )
        else:
            return auc(
                m=self.m,
                xs=self.data.xs,
                y=self.data.y,
                plot=True,
                *args,
                **kwargs,
            )

    def confusion_matrix(self, val: bool = True, *args, **kwargs):
        if val:
            return confusion_matrix(
                m=self.m, xs=self.data.val_xs, y=self.data.val_y, *args, **kwargs
            )
        else:
            return confusion_matrix(
                m=self.m, xs=self.data.xs, y=self.data.y, *args, **kwargs
            )

    def plot_one_way_error(self, col: Optional[str] = None, *args, **kwargs):
        col = ifnone(col, self.data.dep_var)
        return plot_one_way_error(
            df=self.data.df,
            error=self.raw_error["cross_entropy"]["model"]["overall"],
            x_col=col,
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols: List[str], *args, **kwargs):
        return plot_two_way_error(
            df=self.data.df,
            error=self.raw_error["cross_entropy"]["model"]["overall"],
            x_cols=cols,
            *args,
            **kwargs,
        )

    def _get_error(self):
        auc_score = {
            "model": get_error(
                func=auc,
                m=self.m,
                train_xs=self.data.train_xs,
                train_y=self.data.train_y,
                val_xs=self.data.val_xs,
                val_y=self.data.val_y,
                xs=self.data.xs,
                y=self.data.y,
                mean=False,
            ),
        }

        cross_entropy_error = {
            "benchmark": get_benchmark_error(
                func=cross_entropy,
                benchmark=self.benchmark,
                train_y=self.data.train_y,
                val_y=self.data.val_y,
                y=self.data.y,
                mean=True,
            ),
            "model": get_error(
                func=m_cross_entropy,
                m=self.m,
                train_xs=self.data.train_xs,
                train_y=self.data.train_y,
                val_xs=self.data.val_xs,
                val_y=self.data.val_y,
                xs=self.data.xs,
                y=self.data.y,
                mean=True,
            ),
        }

        return {"auc": auc_score, "cross_entropy": cross_entropy_error}

    def _get_raw_error(self):
        cross_entropy_prob = {
            "benchmark": get_benchmark_error(
                func=cross_entropy,
                benchmark=self.benchmark,
                train_y=self.data.train_y,
                val_y=self.data.val_y,
                y=self.data.y,
                mean=False,
            ),
            "model": get_error(
                func=m_cross_entropy,
                m=self.m,
                train_xs=self.data.train_xs,
                train_y=self.data.train_y,
                val_xs=self.data.val_xs,
                val_y=self.data.val_y,
                xs=self.data.xs,
                y=self.data.y,
                mean=False,
            ),
        }
        return {"cross_entropy": cross_entropy_prob}
