import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from modelflow.clean.base import Clean


class CatStrategy:
    "Namespace containing the various categorical encoding strategies."

    def one_hot(df, col, cat_cols, cat_mapping):
        one_hot = pd.get_dummies(df[col])
        one_hot.columns = [f"{col}_{i}" for i in one_hot.columns]
        df = df.drop(col, axis=1)
        df = df.join(one_hot)
        cat_cols.extend(list(one_hot.columns))
        return df, cat_cols, cat_mapping

    def ordinal(df, col, cat_cols, cat_mapping):
        if pd.isnull(df[col]).any():
            df[col] = df[col].fillna("None")
        cat_convert = df[col].astype("category")
        cat_cols.append(col)
        cat_mapping[col] = dict(enumerate(cat_convert.cat.categories))
        df[col] = cat_convert.cat.codes
        return df, cat_cols, cat_mapping


class EncodeCategorical(Clean):
    def __init__(self, cat_strategy="ordinal"):
        self.cat_cols = []
        self.cat_mapping = {}
        self.cat_strategy = self._assign_cat_strategy(cat_strategy)

    def fit(self, df, cat_cols):
        for col in cat_cols:
            df, self.cat_cols, self.cat_mapping = self.cat_strategy(
                df, col, self.cat_cols, self.cat_mapping
            )
        return df

    def fit_transform(self, df, cat_cols):
        return self.fit(df, cat_cols)

    def transform(self, df, cat_cols):
        return self.fit(df, cat_cols)

    def _assign_cat_strategy(self, cat_strategy):
        if callable(cat_strategy):
            return cat_strategy
        else:
            if cat_strategy not in dir(CatStrategy):
                raise ValueError("Categorical encoding strategy not valid")
            return getattr(CatStrategy, cat_strategy)


def encode_list(df, col, include_col_name=True):
    mlb = MultiLabelBinarizer()
    df = df.join(
        pd.DataFrame(
            mlb.fit_transform(df[col]),
            columns=[f"{col}_{i}" for i in mlb.classes_]
            if include_col_name
            else mlb.classes_,
            index=df.index,
        )
    )
    return df
