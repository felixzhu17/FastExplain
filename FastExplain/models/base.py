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
        model: Union[str, type, Callable] = "rf",
        default_models: dict = {},
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

        self.data = prepare_data(
            df=df,
            cat_names=cat_names,
            cont_names=cont_names,
            max_card=max_card,
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

        if isinstance(model, type):
            self.model_fit_func = prepare_model_class(model)
        elif isinstance(model, Callable):
            self.model_fit_func = model
        elif isinstance(model, str):
            if model in default_models:
                self.model_fit_func = default_models[model]
        else:
            raise ValueError(f"Model can only be one of {', '. join(default_models)}")

        if hypertune:
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

    @abstractmethod
    def _get_error(self):
        pass

    @abstractmethod
    def _get_raw_error(self):
        pass
