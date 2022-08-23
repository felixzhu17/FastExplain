from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from FastExplain.clean.encode_categorical import EncodeCategorical
from FastExplain.clean.fill_missing import FillMissing
from FastExplain.clean.shrink import df_shrink
from FastExplain.clean.split import (
    cont_cat_split,
    get_train_val_split_index,
    split_train_val,
)


def prepare_data(
    df: pd.DataFrame,
    dep_var: Optional[str] = None,
    cat_names: Optional[List[str]] = None,
    cont_names: Optional[List[str]] = None,
    max_card: int = 20,
    max_sparsity: float = 0.25,
    perc_train: int = 0.8,
    seed: int = 0,
    splits: Optional[List[List]] = None,
    cat_strategy: str = "ordinal",
    fill_strategy: str = "median",
    fill_const: int = 0,
    na_dummy: bool = True,
    cont_transformations: List[type] = [],
    reduce_memory: bool = False,
    return_class: bool = True,
):
    """
    Cleans data, performing the following steps:

    1. Checks for Classification
    2. Identifies continuous and categorical columns
    3. Encode categorical columns with ordinal or one_hot encoding
    4. Split data with stratification
    5. Fill missing values with median, constant or mode
    6. Apply additional numeric transformations specified by user

    Args:
        df (pd.DataFrame):
            Pandas DataFrame with columns including the dependent and predictor variables.
        dep_var (str):
            Column name of dependent variable. Defaults to None.
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
        return_class (bool, optional):
            Whether to return a class storing cleaning information. Defaults to True.
    """

    pandas_clean = PandasClean(
        df=df,
        dep_var=dep_var,
        cat_names=cat_names,
        cont_names=cont_names,
        max_card=max_card,
        max_sparsity=max_sparsity,
        perc_train=perc_train,
        seed=seed,
        splits=splits,
        cat_strategy=cat_strategy,
        fill_strategy=fill_strategy,
        fill_const=fill_const,
        na_dummy=na_dummy,
        cont_transformations=cont_transformations,
        reduce_memory=reduce_memory,
    )

    if return_class:
        return pandas_clean

    else:
        if perc_train == 0:
            return (
                pandas_clean.train_xs,
                pandas_clean.train_y,
            )
        else:
            return (
                pandas_clean.train_xs,
                pandas_clean.train_y,
                pandas_clean.val_xs,
                pandas_clean.val_y,
            )


class PandasClean:
    """
    Cleans data, performing the following steps:

    1. Checks for Classification
    2. Identifies continuous and categorical columns
    3. Encode categorical columns with ordinal or one_hot encoding
    4. Split data with stratification
    5. Fill missing values with median, constant or mode
    6. Apply additional numeric transformations specified by user

    Args:
        df (pd.DataFrame):
            Pandas DataFrame with columns including the dependent and predictor variables.
        dep_var (str):
            Column name of dependent variable. Defaults to None.
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

    Attributes:
        df:
            Original Pandas DataFrame supplied
        xs:
            DataFrame of cleaned predictor values
        y:
            Array of dependent values
        train_xs:
            DataFrame of training predictor columns
        train_y:
            Array of training dependent values
        val_xs:
            DataFrame of validation predictor columns
        val_y:
            Array of validation dependent values
        cat_names:
            Name of categorical columns
        cont_names:
            Name of continuous columns
        cat_mapping:
            Mapping of categorical values if ordinal encoding was used
        dep_var:
            Name of dependent variable
        perc_train:
            Percentage of data to split as training
        splits:
            Index of training and validation split
        stratify:
            Whether splits were stratified on dependent variable
        seed:
            Random seed used for splitting
        na_dummy:
            Whether dummy column was used for numerical NA values
        transformations:
            List of transformation classes applied to variables.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dep_var: str,
        cat_names: Optional[List[str]] = None,
        cont_names: Optional[List[str]] = None,
        max_card: int = 20,
        max_sparsity: float = 0.25,
        perc_train: int = 0.8,
        seed: int = 0,
        splits: Optional[List[List]] = None,
        cat_strategy: str = "ordinal",
        fill_strategy: str = "median",
        fill_const: int = 0,
        na_dummy: bool = True,
        cont_transformations: List[type] = [],
        reduce_memory: bool = True,
    ):

        self.df = df.reset_index(drop=True)
        self.cat_mapping = {}
        self.dep_var = dep_var
        self.perc_train = perc_train
        self.splits = splits
        self.stratify = None
        self.seed = seed
        self.na_dummy = na_dummy
        self.transformations = {}
        if reduce_memory:
            self.df = df_shrink(self.df, int2uint=True)

        if cat_names is None or cont_names is None:
            cont, cat = cont_cat_split(
                df, max_card=max_card, max_sparsity=max_sparsity, dep_var=self.dep_var
            )

        if cat_names is None:
            self.cat_names = cat.copy()
        else:
            self.cat_names = cat_names

        if cont_names is None:
            self.cont_names = cont.copy()
        else:
            self.cont_names = cont_names

        # Check classification
        if self.dep_var:
            check_dep_var(self.df, self.dep_var)
            self.classification = check_classification(self.df[self.dep_var])
            if self.classification:
                self._prepare_classification()
            else:
                try:
                    self.df[self.dep_var] = self.df[self.dep_var].astype("float")
                except ValueError:
                    raise NotImplementedError("Multi-classification not supported yet")

        # Replace INF with NAN
        self.df = self.df.replace(np.inf, np.nan)

        # Convert column types
        encode_cat = EncodeCategorical(cat_strategy=cat_strategy)
        self.df = encode_cat.fit_transform(self.df, self.cat_names)
        self.cat_names = encode_cat.cat_cols
        self.cat_mapping.update(encode_cat.cat_mapping)
        self._record_transformation(encode_cat)

        # Gather data
        self.xs = self.df[self.cat_names + self.cont_names]
        if self.dep_var:
            self.y = self.df[self.dep_var]

        # Perform splits
        if self.splits is None:
            self.splits = get_train_val_split_index(
                self.df, self.perc_train, seed=self.seed, stratify=self.stratify
            )
        if self.dep_var:
            self.train_xs, self.train_y, self.val_xs, self.val_y = split_train_val(
                xs=self.xs, y=self.y, splits=self.splits
            )
        else:
            (
                self.train_xs,
                self.val_xs,
            ) = split_train_val(xs=self.xs, y=None, splits=self.splits)

        # Fill Missing
        fill_missing = FillMissing(
            fill_strategy=fill_strategy, fill_const=fill_const, na_dummy=na_dummy
        )
        self.train_xs = fill_missing.fit_transform(self.train_xs, self.cont_names)
        self.val_xs = fill_missing.transform(self.val_xs)
        if self.na_dummy:
            self.cat_names.extend(fill_missing.na_dummy_cols)
            self.cat_mapping.update(fill_missing.na_dummy_col_mappings)
        self._record_transformation(fill_missing)

        # Continuous Transformations
        for transform in cont_transformations:
            if isinstance(transform, type):
                transform = transform()
            self._check_transformation(transform)
            self.train_xs[self.cont_names] = transform.fit_transform(
                self.train_xs[self.cont_names]
            )
            self.val_xs[self.cont_names] = transform.transform(
                self.val_xs[self.cont_names]
            )
            self._record_transformation(transform)

        # Gather final data
        self.xs = pd.concat([self.train_xs, self.val_xs])
        if self.dep_var:
            self.y = pd.concat([self.train_y, self.val_y])

    def _prepare_classification(self):
        y_cat = self.df[self.dep_var].astype("category")
        self.cat_mapping[self.dep_var] = dict(enumerate(y_cat.cat.categories))
        self.df[self.dep_var] = y_cat.cat.codes
        self.stratify = self.df[self.dep_var]

    def _record_transformation(self, transform_class: type):
        self.transformations[type(transform_class).__name__] = transform_class

    def _check_transformation(self, transform_class: type):
        if hasattr(transform_class, "fit_transform") and hasattr(
            transform_class, "transform"
        ):
            return
        else:
            raise ValueError(
                f"{transform_class} needs to have fit_transform and transform attributes"
            )


def check_classification(y):

    unique_y = len(y.unique())
    if unique_y == 1:
        raise ValueError("Dependent Variable only has 1 unique value")

    if not is_numeric_dtype(y):
        if unique_y > 2:
            raise NotImplementedError("Multi-class classification not supported yet")

    return unique_y == 2


def check_dep_var(df, dep_var):
    if isinstance(dep_var, str):
        pass
    else:
        raise ValueError("Dependent Variable must be a string")

    if is_numeric_dtype(df[dep_var]):

        if df[dep_var].isna().sum() == 0:
            pass
        else:
            raise ValueError("Dependent Variable has missing values")

        if np.isinf(df[dep_var]).sum() == 0:
            pass
        else:
            raise ValueError("Dependent Variable has inf values")

    return
