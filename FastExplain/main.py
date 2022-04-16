import pandas as pd
from typing import Optional, List, Type
from FastExplain.explain import Explain
from FastExplain.clean import prepare_data, check_classification
from FastExplain.models import (
    rf_reg,
    xgb_reg,
    ebm_reg,
    rf_class,
    xgb_class,
    ebm_class,
    get_model_parameters,
)
from FastExplain.metrics import (
    get_benchmark_error,
    r_mse,
    get_error,
    m_rmse,
    plot_one_way_error,
    plot_two_way_error,
    auc,
    m_cross_entropy,
    confusion_matrix,
)
from FastExplain.utils import root_mean, ifnone

REG_MODELS = {"rf": rf_reg, "xgb": xgb_reg, "ebm": ebm_reg}
CLASS_MODELS = {
    "rf": rf_class,
    "xgb": xgb_class,
    "ebm": ebm_class,
}


def model_data(
    df: pd.DataFrame,
    dep_var: str,
    cat_names: Optional[List[str]] = None,
    cont_names: Optional[List[str]] = None,
    model: str = "rf",
    *args,
    **kwargs,
):
    classification = check_classification(df[dep_var])
    if classification:
        return Classification(
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            model=model,
            *args,
            **kwargs,
        )
    else:
        return Regression(
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            model=model,
            *args,
            **kwargs,
        )


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
        cont_transformations: List[Type] = [],
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
        cont_transformations: List[Type] = [],
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
