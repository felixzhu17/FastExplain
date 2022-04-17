from typing import List, Optional

import pandas as pd

from FastExplain.clean import prepare_data
from FastExplain.explain import Explain
from FastExplain.metrics import (
    get_benchmark_error,
    get_error,
    m_rmse,
    plot_one_way_error,
    plot_two_way_error,
    r_mse,
)
from FastExplain.models.algorithms import ebm_reg, get_model_parameters, rf_reg, xgb_reg
from FastExplain.utils import root_mean

REG_MODELS = {"rf": rf_reg, "xgb": xgb_reg, "ebm": ebm_reg}


class Regression(
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
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
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

        self.benchmark = self.data.train_y.mean()
        if model in REG_MODELS:
            self.m = REG_MODELS[model](
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Model can only be one of {', '. join(REG_MODELS)}")

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

    def plot_one_way_error(self, col: Optional[str] = None, *args, **kwargs):
        col = col if col else self.data.dep_var
        return plot_one_way_error(
            self.data.df,
            self.raw_error["squared_error"]["model"]["overall"],
            col,
            func=root_mean,
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols: List[str], *args, **kwargs):
        return plot_two_way_error(
            self.data.df,
            self.raw_error["squared_error"]["model"]["overall"],
            cols,
            func=root_mean,
            *args,
            **kwargs,
        )

    def _get_error(self):
        rmse = {
            "benchmark": get_benchmark_error(
                r_mse,
                self.benchmark,
                self.data.train_y,
                self.data.val_y,
                self.data.y,
                True,
            ),
            "model": get_error(
                m_rmse,
                self.m,
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                self.data.xs,
                self.data.y,
                True,
            ),
        }
        return {"rmse": rmse}

    def _get_raw_error(self):
        squared_error = {
            "benchmark": get_benchmark_error(
                r_mse,
                self.benchmark,
                self.data.train_y,
                self.data.val_y,
                self.data.y,
                False,
            ),
            "model": get_error(
                m_rmse,
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
        return {"squared_error": squared_error}
