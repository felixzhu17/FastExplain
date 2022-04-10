import pandas as pd
from .encode_categorical import EncodeCategorical
from .fill_missing import FillMissing
from .shrink import df_shrink
from .split import get_train_val_split_index, split_train_val


def prepare_data(
    df,
    cat_names=None,
    cont_names=None,
    dep_var=None,
    perc_train=0.8,
    seed=0,
    splits=None,
    cat_strategy="ordinal",
    fill_strategy="median",
    fill_const=0,
    na_dummy=True,
    cont_transformations=[],
    reduce_memory=True,
    return_class=False,
):
    pandas_clean = PandasClean(
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
    def __init__(
        self,
        df,
        cat_names=None,
        cont_names=None,
        dep_var=None,
        perc_train=0.8,
        seed=0,
        splits=None,
        cat_strategy="ordinal",
        fill_strategy="median",
        fill_const=0,
        na_dummy=True,
        cont_transformations=[],
        reduce_memory=True,
    ):
        self.df = df.reset_index(drop=True)
        self.cat_names = cat_names.copy()
        self.cont_names = cont_names.copy()
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

        # Check classification
        if self.dep_var:
            self.classification = check_classification(self.df[self.dep_var])
            if self.classification:
                self._prepare_classification()
            else:
                try:
                    self.df[self.dep_var] = self.df[self.dep_var].astype("float")
                except ValueError:
                    raise NotImplementedError("Multi-classification not supported yet")

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
        self.train_xs, self.train_y, self.val_xs, self.val_y = split_train_val(
            self.xs, self.y, self.splits
        )

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

        self.df = self.xs.assign(**{dep_var: self.y})

    def _prepare_classification(self):
        y_cat = self.df[self.dep_var].astype("category")
        self.cat_mapping[self.dep_var] = dict(enumerate(y_cat.cat.categories))
        self.df[self.dep_var] = y_cat.cat.codes
        self.stratify = self.df[self.dep_var]

    def _record_transformation(self, transform_class):
        self.transformations[type(transform_class).__name__] = transform_class


def check_classification(y):
    unique_y = len(y.unique())
    if unique_y == 1:
        raise ValueError("Dependent Variable only has 1 unique value")
    return unique_y == 2
