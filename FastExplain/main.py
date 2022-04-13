from FastExplain.explain import Explain
from FastExplain.clean import prepare_data, check_classification
from FastExplain.models import rf_reg, xgb_reg, ebm_reg, rf_class, xgb_class, ebm_class, get_model_parameters
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
from FastExplain.utils import root_mean

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
    model="rf",
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
            model=model,
            *args,
            **kwargs,
        )
    else:
        return Regression(
            df,
            cat_names,
            cont_names,
            dep_var,
            model=model,
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

        self.rmse = {
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

        self.squared_error = {
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
        
        self.params = get_model_parameters(self.m, model)

        Explain.__init__(
            self,
            self.m,
            self.data.xs,
            self.data.df,
            self.data.dep_var,
            self.data.train_xs.columns,
        )

    def plot_one_way_error(self, col=None, *args, **kwargs):
        col = col if col else self.data.dep_var
        return plot_one_way_error(
            self.data.df,
            self.squared_error["model"]["overall"],
            col,
            func=root_mean,
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols, *args, **kwargs):
        return plot_two_way_error(
            self.data.df,
            self.squared_error["model"]["overall"],
            cols,
            func=root_mean,
            *args,
            **kwargs,
        )


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
                self.data.train_xs,
                self.data.train_y,
                self.data.val_xs,
                self.data.val_y,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Model can only be one of {', '. join(CLASS_MODELS)}")

        self.auc = {
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

        self.cross_entropy = {
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

        self.cross_entropy_prob = {
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
        
        self.params = get_model_parameters(self.m, model)

        Explain.__init__(
            self,
            self.m,
            self.data.xs,
            self.data.df,
            self.data.dep_var,
            self.data.train_xs.columns,
        )

    def plot_auc(self, val=True, *args, **kwargs):
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

    def confusion_matrix(self, val=True, *args, **kwargs):
        if val:
            return confusion_matrix(
                self.m, self.data.val_xs, self.data.val_y, *args, **kwargs
            )
        else:
            return confusion_matrix(self.m, self.data.xs, self.data.y, *args, **kwargs)

    def plot_one_way_error(self, col=None, *args, **kwargs):
        col = col if col else self.data.dep_var
        return plot_one_way_error(
            self.data.df,
            self.cross_entropy_prob["model"]["overall"],
            col,
            *args,
            **kwargs,
        )

    def plot_two_way_error(self, cols, *args, **kwargs):
        return plot_two_way_error(
            self.data.df,
            self.cross_entropy_prob["model"]["overall"],
            cols,
            *args,
            **kwargs,
        )
