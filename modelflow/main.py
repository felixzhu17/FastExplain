from .explain import Explain
from .clean import prepare_data, check_classification
from .models import *
from .metrics import *

REG_MODELS = {"rf": rf_reg, "xgb": xgb_reg, "ebm": ebm_reg}
CLASS_MODELS = {
    "rf": rf_class,
    "xgb": xgb_class,
    "ebm": ebm_class,
}


def model_data(
    df,
    cat_names,
    cont_names,
    dep_var,
    *args,
    **kwargs,
):
    classification = check_classification(df[dep_var])
    if classification:
        return Classification(
            df,
            cat_names,
            cont_names,
            dep_var,
            *args,
            **kwargs,
        )
    else:
        return Regression(
            df,
            cat_names,
            cont_names,
            dep_var,
            *args,
            **kwargs,
        )


class Regression(
    Explain,
):
    def __init__(
        self,
        df,
        cat_names,
        cont_names,
        dep_var,
        model="rf",
        perc_train=0.8,
        seed=0,
        splits=None,
        cat_strategy="ordinal",
        fill_strategy="median",
        fill_const=0,
        na_dummy=True,
        cont_transformations=[],
        reduce_memory=True,
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

        self.benchmark = self.data.train_y.mean()
        self.benchmark_rmse = r_mse(self.benchmark, self.data.train_y), r_mse(
            self.benchmark, self.data.val_y
        )

        if model in REG_MODELS:
            self.m = REG_MODELS[model](
                self.data.train_xs, self.data.train_y, *args, **kwargs
            )
        else:
            raise ValueError(f"Model can only be one of {', '. join(REG_MODELS)}")

        self.model_rmse = m_rmse(self.m, self.data.train_xs, self.data.train_y), m_rmse(
            self.m, self.data.val_xs, self.data.val_y
        )

        Explain.__init__(self, self.m, self.data.xs, self.data.df, self.data.dep_var)


class Classification(
    Explain,
):
    def __init__(
        self,
        df,
        cat_names,
        cont_names,
        dep_var,
        model="rf",
        perc_train=0.8,
        seed=0,
        splits=None,
        cat_strategy="ordinal",
        fill_strategy="median",
        fill_const=0,
        na_dummy=True,
        cont_transformations=[],
        reduce_memory=True,
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
                self.data.train_xs, self.data.train_y, *args, **kwargs
            )
        else:
            raise ValueError(f"Model can only be one of {', '. join(CLASS_MODELS)}")

        Explain.__init__(self, self.m, self.data.xs, self.data.df, self.data.dep_var)

    def plot_roc_curve(self, *args, **kwargs):
        return plot_roc_curve(
            self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
        )

    def confusion_matrix(self, *args, **kwargs):
        return confusion_matrix(
            self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
        )
