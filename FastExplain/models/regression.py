from typing import Callable, List, Optional, Union

import pandas as pd

from FastExplain.metrics import (
    get_benchmark_error,
    get_error,
    m_rmse,
    plot_one_way_error,
    plot_two_way_error,
    r_mse,
)
from FastExplain.models.algorithms import ebm_reg, rf_reg, xgb_reg
from FastExplain.models.base import Model
from FastExplain.utils import ifnone, root_mean

REG_MODELS = {"rf": rf_reg, "xgb": xgb_reg, "ebm": ebm_reg}


class Regression(
    Model,
):
    """
    Performs the full Regression modelling pipeling returning a class that is a connected interface for all data, models and related explanatory methods. Out of the box, it will:

    1. Identifies continuous and categorical columns
    2. Encode categorical columns with ordinal or one_hot encoding
    3. Split data
    4. Fill missing values with median, constant or mode
    5. Apply additional numeric transformations specified by user
    6. Fit a Random Forest, Explainable Boosting Machine, XGBoost or other model specified by user
    7. Optionally, hypertune model using Bayesian Optimization

    Args:
        df (pd.DataFrame):
            Pandas DataFrame with columns including the dependent and predictor variables.
        dep_var (str):
            Column name of dependent variable
        cat_names (Optional[List[str]], optional):
            Column names of categorical predictor variables. If both cat_names and cont_names is None, they are automatically extracted based on max_card. Defaults to None.
        cont_names (Optional[List[str]], optional):
            Column names of continuous predictor variables. If both cat_names and cont_names is None, they are automatically extracted based on max_card. Defaults to None.
        max_card (int, optional):
            Maximum number of unique values for categorical variable. Defaults to 20.
        max_sparsity (float, optional):
            Maximum number of unique values for categorical variable as proportion of number of rows. Defaults to 0.25.
        model (Union[str, type, Callable], optional):
            Model to fit. Can choose from 'rf', 'xgb' and 'ebm' as defaults, or can provide own model class with fit and predict attributes. Defaults to "ebm".
        perc_train (int, optional):
            Percentage of data to split as training. Defaults to 0.8.
        seed (int, optional):
            Seed for splitting data. Defaults to 0.
        splits (Optional[List[List]], optional):
            Index of df to split data on. If not specified, data will be split based on perc_train. Defaults to None.
        cat_strategy (str, optional):
            Categorical encoding strategy. Can pick from 'ordinal' or 'one_hot'. Defaults to "ordinal".
        fill_strategy (str, optional):
            NA filling strategy. Can pick from 'median', 'constant' or 'mode' . Defaults to "median".
        fill_const (int, optional):
            Value to fill NAs with if fill_strategy is 'constant'. Defaults to 0.
        na_dummy (bool, optional):
            Whether to create a dummy column for NA values. Defaults to True.
        cont_transformations (List[type], optional):
            Additional transformations for continuous predictor variables. Transformations must be supplied as a class with fit_transform and transform attributes. Defaults to [].
        reduce_memory (bool, optional):
            Whether to reduce the memory of df in storage. Defaults to False.
        hypertune (bool, optional):
            Whether to hypertune model with Bayesian Optimization. Defaults to False.
        hypertune_max_evals (int, optional):
            Number of evaluations to hypertune. Defaults to 100.
        hypertune_params (Optional[dict], optional):
            Dictionary containing parameters of model to hypertune. Key for each parameter must be an array of [lower (numeric), upper (numeric), is_integer (bool)]. Defaults to None.
        hypertune_loss_metric (Optional[Callable], optional):
            Function to minimise when hypertuning. Arguments of function must contain m (the model), xs (predictors) and y (target). If not specified, mean squared error is used. Defaults to None.
        use_fitted_model (bool, optional):
            Whether to supply a model that is already fitted. Defaults to False.
        *model_args, **model_kwargs:
            Additional arguments for the model

    Attributes:
        data:
            Class containing data used for modelling. (See FastExplain.clean.main.PandasClean)
        m:
            Fitted model
        params:
            Parameters of the fitted model
        model_fit_func:
            Function used to fit the data and return model
        hypertune:
            If hypertuned, class containing hypertuning information (See FastExplain.models.algorithms.hypertuning.Hypertune)
        benchmark:
            Average of training dependent variables, used as benchmark for fit metrics
        error:
            Root Mean Squared error on training and validation for benchmark and model
        error_raw:
            Root Mean Squared error per observation on training and validation for benchmark and model
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dep_var: str,
        cat_names: Optional[List[str]] = None,
        cont_names: Optional[List[str]] = None,
        max_card: int = 20,
        max_sparsity: float = 0.25,
        model: Union[str, type, Callable] = "ebm",
        perc_train: int = 0.8,
        seed: int = 0,
        splits: Optional[List[List]] = None,
        cat_strategy: str = "ordinal",
        fill_strategy: str = "median",
        fill_const: int = 0,
        na_dummy: bool = True,
        cont_transformations: List[type] = [],
        reduce_memory: bool = False,
        hypertune: bool = False,
        hypertune_max_evals: int = 100,
        hypertune_params: Optional[dict] = None,
        hypertune_loss_metric: Optional[Callable] = None,
        use_fitted_model: bool = False,
        *model_args,
        **model_kwargs,
    ):
        self.classification = False
        Model.__init__(
            self,
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            max_card=max_card,
            max_sparsity=max_sparsity,
            model=model,
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
            use_fitted_model=use_fitted_model,
            default_models=REG_MODELS,
            *model_args,
            **model_kwargs,
        )

    def plot_one_way_error(self, col: Optional[str] = None, *args, **kwargs):
        col = ifnone(col, self.data.dep_var)
        print(col)
        return plot_one_way_error(
            df=self.data.df,
            error=self.raw_error["squared_error"]["model"]["overall"],
            x_col=col,
            func=root_mean,
            yaxis_title="Root Squared Error",
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols: List[str], *args, **kwargs):
        return plot_two_way_error(
            df=self.data.df,
            error=self.raw_error["squared_error"]["model"]["overall"],
            x_cols=cols,
            func=root_mean,
            dep_name="Root Squared Error",
            *args,
            **kwargs,
        )

    def _get_error(self):
        rmse = {
            "benchmark": get_benchmark_error(
                func=r_mse,
                benchmark=self.benchmark,
                train_y=self.data.train_y,
                val_y=self.data.val_y,
                y=self.data.y,
                mean=True,
            ),
            "model": get_error(
                func=m_rmse,
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
        return {"rmse": rmse}

    def _get_raw_error(self):
        squared_error = {
            "benchmark": get_benchmark_error(
                func=r_mse,
                benchmark=self.benchmark,
                train_y=self.data.train_y,
                val_y=self.data.val_y,
                y=self.data.y,
                mean=False,
            ),
            "model": get_error(
                func=m_rmse,
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
        return {"squared_error": squared_error}
