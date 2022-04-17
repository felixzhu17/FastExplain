from typing import List, Optional

import pandas as pd

from FastExplain.clean import check_classification, check_dep_var
from FastExplain.models import Classification, Regression


def model_data(
    df: pd.DataFrame,
    dep_var: str,
    cat_names: Optional[List[str]] = None,
    cont_names: Optional[List[str]] = None,
    model: str = "rf",
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
