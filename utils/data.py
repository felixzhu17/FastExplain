import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from functools import reduce


class DataUtils:
    def quantile_ied(self, x_vec, q):
        """
        Inverse of empirical distribution function (quantile R type 1).

        More details in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles.html
        https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
        https://en.wikipedia.org/wiki/Quantile

        Arguments:
        x_vec -- A pandas series containing the values to compute the quantile for
        q -- An array of probabilities (values between 0 and 1)
        """

        x_vec = x_vec.sort_values()
        n = len(x_vec) - 1
        m = 0
        j = (n * q + m).astype(int)  # location of the value
        g = n * q + m - j

        gamma = (g != 0).astype(int)
        quant_res = (1 - gamma) * x_vec.shift(1, fill_value=0).iloc[
            j
        ] + gamma * x_vec.iloc[j]
        quant_res.index = q
        # add min at quantile zero and max at quantile one (if needed)
        if 0 in q:
            quant_res.loc[0] = x_vec.min()
        if 1 in q:
            quant_res.loc[1] = x_vec.max()
        return quant_res

    def get_bins(self, x, grid_size):

        quantiles = np.append(
            0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size)
        )
        bins = [x.min()] + self.quantile_ied(x, quantiles).to_list()
        return np.unique(bins)

    def percent_cat_agg(self, series, top=None):
        x = Counter(series)
        sum_x = sum(x.values())
        x_perc = {k: v / sum_x for k, v in x.items()}
        if top:
            return Counter(x_perc).most_common(top)
        else:
            return x_perc

    def encode_list(self, df, col, include_col_name=True):
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

    def merge_multi_df(self, dfs, *args, **kwargs):
        return reduce(lambda left, right: pd.merge(left, right, *args, **kwargs), dfs)

    def get_date_freq(self, df, col, freq):
        return df.groupby(df[col].dt.to_period(freq)).count()[col]
