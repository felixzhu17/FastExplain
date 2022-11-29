from typing import Callable, List, Optional, Union

import pandas as pd

from FastExplain.clean import check_classification, check_dep_var
from FastExplain.models import Classification, Regression


def model_data(
    df: pd.DataFrame,
    dep_var: str,
    cat_names: Optional[List[str]] = None,
    cont_names: Optional[List[str]] = None,
    max_card: int = 20,
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
    """
    Performs the full modelling pipeling returning a class that is a connected interface for all data, models and related explanatory methods. Out of the box, it will:

    1. Checks for Classification
    2. Identifies continuous and categorical columns
    3. Encode categorical columns with ordinal or one_hot encoding
    4. Split data with stratification
    5. Fill missing values with median, constant or mode
    6. Apply additional numeric transformations specified by user
    7. Fit a Random Forest, Explainable Boosting Machine, XGBoost or other model specified by user
    8. Optionally, hypertune model using Bayesian Optimization

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
            Function to minimise when hypertuning. Arguments of function must contain m (the model), xs (predictors) and y (target). If not specified, cross entropy is used for classification and mean squared errr for regression. Defaults to None.
        use_fitted_model (bool, optional):
            Whether to supply a model that is already fitted. Defaults to False.
        *model_args, **model_kwargs:
            Additional arguments for the model
    """

    check_dep_var(df, dep_var)
    classification = check_classification(df[dep_var])
    if classification:
        return Classification(
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            max_card=max_card,
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
            *model_args,
            **model_kwargs,
        )
    else:
        return Regression(
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
            max_card=max_card,
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
            *model_args,
            **model_kwargs,
        )
