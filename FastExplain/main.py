from typing import Callable, List, Optional, Union

import pandas as pd

from FastExplain.clean import check_classification, check_dep_var
from FastExplain.models import Classification, Regression


def model_data(
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
    *args,
    **kwargs,
):
    check_dep_var(dep_var)
    classification = check_classification(df[dep_var])
    if classification:
        return Classification(
            df=df,
            dep_var=dep_var,
            cat_names=cat_names,
            cont_names=cont_names,
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
            perc_train=perc_train,
            seed=seed,
            splits=splits,
            cat_strategy=cat_strategy,
            fill_strategy=fill_strategy,
            fill_const=fill_const,
            na_dummy=na_dummy,
            cont_transformations=cont_transformations,
            reduce_memory=reduce_memory,
            *args,
            **kwargs,
        )
