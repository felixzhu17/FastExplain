import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import pandas as pd

from FastExplain.clean import prepare_data
from FastExplain.explain import Explain
from FastExplain.models.algorithms import (
    Hypertune,
    get_default_hypertune_params,
    get_model_parameters,
    prepare_model_class,
)
from FastExplain.utils import ifnone


class Model(Explain, ABC):
    """
    Base class for full modelling pipeline
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
        default_models: dict = {},
        *model_args,
        **model_kwargs,
    ):

        self._data_prep_args = dict(
            cat_names=cat_names,
            cont_names=cont_names,
            max_card=max_card,
            max_sparsity=max_sparsity,
            dep_var=dep_var,
            cat_strategy=cat_strategy,
            fill_strategy=fill_strategy,
            fill_const=fill_const,
            na_dummy=na_dummy,
            cont_transformations=cont_transformations,
            reduce_memory=reduce_memory,
        )

        self.data = self.prepare_data(df, perc_train, seed, splits)
        if use_fitted_model:
            self.m = model
        else:
            if isinstance(model, type):
                self.model_fit_func = prepare_model_class(model)
            elif isinstance(model, Callable):
                self.model_fit_func = model
            elif isinstance(model, str):
                if model in default_models:
                    self.model_fit_func = default_models[model]
            else:
                raise ValueError(
                    f"Model can only be one of {', '. join(default_models)}"
                )

            if hypertune and model != "ebm":
                hypertune_params = ifnone(
                    hypertune_params, get_default_hypertune_params(model)
                )
                self.hypertune = Hypertune(
                    xs=self.data.train_xs,
                    y=self.data.train_y,
                    val_xs=self.data.val_xs,
                    val_y=self.data.val_y,
                    model_fit_func=self.model_fit_func,
                    hypertune_loss_metric=hypertune_loss_metric,
                )
                self.m = self.hypertune.hypertune_model(
                    hypertune_params=hypertune_params,
                    hypertune_max_evals=hypertune_max_evals,
                    *model_args,
                    **model_kwargs,
                )
            else:
                if model == "ebm":
                    warnings.warn("Hypertuning not implemented for EBM")
                self.m = self.model_fit_func(
                    self.data.train_xs,
                    self.data.train_y,
                    *model_args,
                    **model_kwargs,
                )

        self.benchmark = self.data.train_y.mean()
        self.error = self._get_error()
        self.raw_error = self._get_raw_error()
        self.params = get_model_parameters(self.m)

        Explain.__init__(
            self,
            self.m,
            self.data.xs,
            self.data.df,
            self.data.dep_var,
            self.data.cat_mapping,
        )

    def prepare_data(
        self,
        df,
        perc_train: int = 0.8,
        seed: int = 0,
        splits: Optional[List[List]] = None,
        return_class=True,
    ):
        return prepare_data(
            df=df,
            **self._data_prep_args,
            return_class=return_class,
            perc_train=perc_train,
            seed=seed,
            splits=splits,
        )

    @abstractmethod
    def _get_error(self):
        pass

    @abstractmethod
    def _get_raw_error(self):
        pass
