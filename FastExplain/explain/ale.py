import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

from FastExplain.clean import check_cont_col
from FastExplain.explain.bin import CI_estimate, quantile_ied
from FastExplain.utils import (
    COLOURS,
    bin_columns,
    clean_text,
    cycle_colours,
    fill_list,
    get_upper_lower_bound_traces,
    ifnone,
    merge_multi_df,
    plot_two_way,
    plot_upper_lower_bound_traces,
)


def ale_summary(m, xs, col, model_names=None, *args, **kwargs):
    if isinstance(m, (list, tuple)):
        model_names = ifnone(model_names, [f"Model {i}" for i in range(len(m))])
        ales = []
        for count, ale_info in enumerate(zip(m, xs)):
            model, x_values = ale_info
            if count == len(m) - 1:
                ales.append(
                    _clean_ale(model, x_values, col, *args, **kwargs)[["eff", "size"]]
                )
            else:
                ales.append(_clean_ale(model, x_values, col, *args, **kwargs)[["eff"]])

        output = merge_multi_df(ales, left_index=True, right_index=True)
        output.columns = model_names + ["size"]
        return output
    else:
        return _clean_ale(m, xs, col, *args, **kwargs)


def _clean_ale(
    m,
    xs,
    col,
    numeric=None,
    normalize=True,
    percentage=False,
    condense_last=True,
    remove_last_bins=None,
    dp=2,
    filter=None,
    bins=None,
    *args,
    **kwargs,
):
    xs = xs.query(filter) if filter else xs
    numeric = ifnone(numeric, check_cont_col(xs[col]))
    bins = bins if numeric else sorted(list(xs[col].unique()))
    df = aleplot_1D_continuous(xs, model=m, feature=col, bins=bins, *args, **kwargs)
    df = df[~df.index.duplicated(keep="last")]
    adjust = -1 * df.iloc[0]["eff"]
    df["eff"] += adjust
    df["lowerCI_95%"] += adjust
    df["upperCI_95%"] += adjust

    if numeric is False:
        df["size"] = list(xs[col].value_counts().sort_index())

    if normalize and numeric:
        df.index = convert_ale_index(
            pd.to_numeric(df.index), dp, percentage, condense_last
        )
    if remove_last_bins:
        df = df.iloc[:-remove_last_bins]
    return df


def plot_ale(
    m,
    xs,
    col,
    dep_name=None,
    feature_name=None,
    model_names=None,
    plotsize=None,
    *args,
    **kwargs,
):

    feature_name = ifnone(feature_name, clean_text(col))
    if isinstance(m, (list, tuple)):
        model_names = (
            model_names if model_names else [f"Model {i}" for i in range(len(m))]
        )
        for count, ale_info in enumerate(zip(m, xs, model_names, cycle_colours())):
            model, x_values, model_name, color = ale_info
            if count == 0:
                traces, x, size = _get_ale_traces(
                    model,
                    x_values,
                    col,
                    model_name,
                    color,
                    return_index_size=True,
                    *args,
                    **kwargs,
                )
            else:
                traces.extend(
                    _get_ale_traces(
                        model,
                        x_values,
                        col,
                        model_name,
                        color,
                        return_index_size=False,
                        *args,
                        **kwargs,
                    )
                )
    else:
        traces, x, size = _get_ale_traces(
            m,
            xs,
            col,
            feature_name,
            COLOURS["blue"],
            return_index_size=True,
            *args,
            **kwargs,
        )

    return plot_upper_lower_bound_traces(
        traces,
        x,
        size,
        x_axis_title=feature_name,
        y_axis_title=dep_name,
        plotsize=plotsize,
    )


def _get_ale_traces(
    m, xs, col, model_name, color, return_index_size=True, *args, **kwargs
):
    df = ale_summary(m, xs, col, *args, **kwargs)
    x = df.index
    y = df["eff"]
    size = df["size"]
    y_lower = df["lowerCI_95%"]
    y_upper = df["upperCI_95%"]
    return get_upper_lower_bound_traces(
        x, y, y_lower, y_upper, size, color, model_name, return_index_size
    )


def plot_multi_ale(m, xs, cols, index, plotsize=None, *args, **kwargs):
    pdp = {
        i: fill_list(list(ale_summary(m, xs, i, *args, **kwargs)["eff"]), len(index))
        for i in cols
    }
    pdp_df = pd.DataFrame(pdp, index=index)
    fig = px.line(pdp_df, x=pdp_df.index, y=pdp_df.columns)
    if plotsize:
        fig.update_layout(
            width=plotsize[0],
            height=plotsize[1],
        )
    fig.update_layout(plot_bgcolor="white")
    return fig


def plot_2d_ale(
    m,
    xs,
    cols,
    dp=2,
    feature_names=None,
    percentage=False,
    condense_last=True,
    plotsize=None,
    colorscale="Blues",
    *args,
    **kwargs,
):
    df = aleplot_2D_continuous(xs, m, cols, *args, **kwargs)
    df = df - df.min().min()
    df.index = convert_ale_index(df.index, dp, percentage, condense_last)
    df.columns = convert_ale_index(df.columns, dp, percentage, condense_last)
    return plot_two_way(df, cols, feature_names, plotsize, colorscale)


def convert_ale_index(index, dp, percentage, condense_last):
    if percentage:
        return [f"{index[0]:,.{dp}%}"] + bin_columns(
            index, dp=dp, percentage=percentage, condense_last=condense_last
        )
    else:
        return [f"{index[0]:,.{dp}f}"] + bin_columns(
            index, dp=dp, percentage=percentage, condense_last=condense_last
        )


class Ale:
    def __init__(self, m, xs, dep_var=None):
        self.m = m
        self.xs = xs
        self.dep_var = dep_var

    def ale_summary(self, *args, **kwargs):
        return ale_summary(self.m, self.xs, *args, **kwargs)

    def plot_ale(self, *args, **kwargs):
        return plot_ale(self.m, self.xs, dep_name=self.dep_var, *args, **kwargs)

    def plot_multi_ale(self, *args, **kwargs):
        return plot_multi_ale(self.m, self.xs, *args, **kwargs)

    def plot_2d_ale(self, *args, **kwargs):
        return plot_2d_ale(self.m, self.xs, *args, **kwargs)


def aleplot_1D_continuous(
    X, model, feature, grid_size=20, bins=None, include_CI=True, C=0.95
):
    """Compute the accumulated local effect of a numeric continuous feature.

    This function divides the feature in question into grid_size intervals (bins)
    and computes the difference in prediction between the first and last value
    of each interval and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    feature -- String, the name of the column holding the feature being studied.
    grid_size -- An integer indicating the number of intervals into which the
    feature range is divided.
    bins -- Bins indicating what intervals the feature range is divided
    include_CI -- A boolean, if True the confidence interval
    of the effect is returned with the results.
    C -- A float the confidence level for which to compute the confidence interval

    Return: A pandas DataFrame containing for each bin: the size of the sample in it
    and the accumulated centered effect of this bin.
    """

    if bins is None:
        quantiles = np.append(
            0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size)
        )
        # use customized quantile function to get the same result as
        # type 1 R quantile (Inverse of empirical distribution function)
        bins = [X[feature].min()] + quantile_ied(X[feature], quantiles).to_list()

    bins = np.unique(bins)
    feat_cut = pd.cut(X[feature], bins, include_lowest=True)
    bin_codes = feat_cut.cat.codes
    bin_codes_unique = np.unique(bin_codes)

    X1 = X.copy()
    X2 = X.copy()
    X1[feature] = [bins[i] for i in bin_codes]
    X2[feature] = [bins[i + 1] for i in bin_codes]
    try:
        y_1 = model.predict(X1).ravel()
        y_2 = model.predict(X2).ravel()
    except Exception as ex:
        raise Exception(
            "Please check that your model is fitted, and accepts X as input."
        )

    delta_df = pd.DataFrame({feature: bins[bin_codes + 1], "Delta": y_2 - y_1})
    res_df = delta_df.groupby([feature]).Delta.agg([("eff", "mean"), "size"])
    res_df["eff"] = res_df["eff"].cumsum()
    res_df.loc[min(bins), :] = 0
    # subtract the total average of a moving average of size 2
    mean_mv_avg = (
        (res_df["eff"] + res_df["eff"].shift(1, fill_value=0)) / 2 * res_df["size"]
    ).sum() / res_df["size"].sum()
    res_df = res_df.sort_index().assign(eff=res_df["eff"] - mean_mv_avg)
    if include_CI:
        ci_est = delta_df.groupby(feature).Delta.agg(
            [("CI_estimate", lambda x: CI_estimate(x, C=C))]
        )
        ci_est = ci_est.sort_index()
        lowerCI_name = "lowerCI_" + str(int(C * 100)) + "%"
        upperCI_name = "upperCI_" + str(int(C * 100)) + "%"
        res_df[lowerCI_name] = res_df[["eff"]].subtract(ci_est["CI_estimate"], axis=0)
        res_df[upperCI_name] = upperCI = res_df[["eff"]].add(
            ci_est["CI_estimate"], axis=0
        )
    return res_df


def aleplot_2D_continuous(X, model, features, grid_size=40, bins=None):
    """Compute the two dimentional accumulated local effect of a two numeric continuous features.

    This function divides the space of the two features into a grid of size
    grid_size*grid_size and computes the difference in prediction between the four
    edges (corners) of each bin in the grid and then centers the results.

    Arguments:
    X -- A pandas DataFrame to pass to the model for prediction.
    model -- Any python model with a predict method that accepts X as input.
    features -- A list of twos strings indicating the names of the columns
    holding the features in question.
    grid_size -- An integer indicating the number of intervals into which each
    feature range is divided.
    bins -- Bins indicating what intervals the feature range is divided

    Return: A pandas DataFrame containing for each bin in the grid
    the accumulated centered effect of this bin.
    """

    # reset index to avoid index missmatches when replacing the values with the codes (lines 50 - 73)
    X = X.reset_index(drop=True)

    if bins is None:
        quantiles = np.append(
            0, np.arange(1 / grid_size, 1 + 1 / grid_size, 1 / grid_size)
        )
        bins_0 = [X[features[0]].min()] + quantile_ied(
            X[features[0]], quantiles
        ).to_list()
        bins_0 = np.unique(bins_0)
        bins_1 = [X[features[1]].min()] + quantile_ied(
            X[features[1]], quantiles
        ).to_list()
        bins_1 = np.unique(bins_1)

    else:
        bins_0, bins_1 = bins

    feat_cut_0 = pd.cut(X[features[0]], bins_0, include_lowest=True)
    bin_codes_0 = feat_cut_0.cat.codes
    bin_codes_unique_0 = np.unique(bin_codes_0)
    feat_cut_1 = pd.cut(X[features[1]], bins_1, include_lowest=True)
    bin_codes_1 = feat_cut_1.cat.codes
    bin_codes_unique_1 = np.unique(bin_codes_1)

    X11 = X.copy()
    X12 = X.copy()
    X21 = X.copy()
    X22 = X.copy()

    X11[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i] for i in bin_codes_0],
            features[1]: [bins_1[i] for i in bin_codes_1],
        }
    )
    X12[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i] for i in bin_codes_0],
            features[1]: [bins_1[i + 1] for i in bin_codes_1],
        }
    )
    X21[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i + 1] for i in bin_codes_0],
            features[1]: [bins_1[i] for i in bin_codes_1],
        }
    )
    X22[features] = pd.DataFrame(
        {
            features[0]: [bins_0[i + 1] for i in bin_codes_0],
            features[1]: [bins_1[i + 1] for i in bin_codes_1],
        }
    )

    y_11 = model.predict(X11).ravel()
    y_12 = model.predict(X12).ravel()
    y_21 = model.predict(X21).ravel()
    y_22 = model.predict(X22).ravel()

    delta_df = pd.DataFrame(
        {
            features[0]: bin_codes_0 + 1,
            features[1]: bin_codes_1 + 1,
            "Delta": (y_22 - y_21) - (y_12 - y_11),
        }
    )
    index_combinations = pd.MultiIndex.from_product(
        [bin_codes_unique_0 + 1, bin_codes_unique_1 + 1], names=features
    )

    delta_df = delta_df.groupby([features[0], features[1]]).Delta.agg(["size", "mean"])

    sizes_df = delta_df["size"].reindex(index_combinations, fill_value=0)
    sizes_0 = sizes_df.groupby(level=0).sum().reindex(range(len(bins_0)), fill_value=0)
    sizes_1 = sizes_df.groupby(level=1).sum().reindex(range(len(bins_0)), fill_value=0)

    eff_df = delta_df["mean"].reindex(index_combinations, fill_value=np.nan)

    # ============== fill in the effects of missing combinations ================= #
    # ============== use the kd-tree nearest neighbour algorithm ================= #
    row_na_idx = np.where(eff_df.isna())[0]
    feat0_code_na = eff_df.iloc[row_na_idx].index.get_level_values(0)
    feat1_code_na = eff_df.iloc[row_na_idx].index.get_level_values(1)

    row_notna_idx = np.where(eff_df.notna())[0]
    feat0_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(0)
    feat1_code_notna = eff_df.iloc[row_notna_idx].index.get_level_values(1)

    if len(row_na_idx) != 0:
        range0 = bins_0.max() - bins_0.min()
        range1 = bins_1.max() - bins_1.min()

        feats_at_na = pd.DataFrame(
            {
                features[0]: (bins_0[feat0_code_na - 1] + bins_0[feat0_code_na])
                / (2 * range0),
                features[1]: (bins_1[feat1_code_na - 1] + bins_1[feat1_code_na])
                / (2 * range1),
            }
        )
        feats_at_notna = pd.DataFrame(
            {
                features[0]: (bins_0[feat0_code_notna - 1] + bins_0[feat0_code_notna])
                / (2 * range0),
                features[1]: (bins_1[feat1_code_notna - 1] + bins_1[feat1_code_notna])
                / (2 * range1),
            }
        )
        # fit the algorithm with the features where the effect is not missing
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(feats_at_notna)
        # find the neighbours of the features where the effect is missing
        distances, indices = nbrs.kneighbors(feats_at_na)
        # fill the missing effects with the effects of the nearest neighbours
        eff_df.iloc[row_na_idx] = eff_df.iloc[
            row_notna_idx[indices.flatten()]
        ].to_list()

    # ============== cumulative sum of the difference ================= #
    eff_df = eff_df.groupby(level=0).cumsum().groupby(level=1).cumsum()

    # ============== centering with the moving average ================= #
    # subtract the cumulative sum of the mean of 1D moving average (for each axis)
    eff_df_0 = eff_df - eff_df.groupby(level=1).shift(periods=1, axis=0, fill_value=0)
    fJ0 = (
        (
            sizes_df
            * (
                eff_df_0.groupby(level=0).shift(periods=1, axis=0, fill_value=0)
                + eff_df_0
            )
            / 2
        )
        .sum(level=0)
        .div(sizes_0)
        .fillna(0)
        .cumsum()
    )

    eff_df_1 = eff_df - eff_df.groupby(level=0).shift(periods=1, axis=0, fill_value=0)
    fJ1 = (
        (
            sizes_df
            * (eff_df_1.groupby(level=1).shift(periods=1, fill_value=0) + eff_df_1)
            / 2
        )
        .sum(level=1)
        .div(sizes_1)
        .fillna(0)
        .cumsum()
    )

    all_combinations = pd.MultiIndex.from_product(
        [[x for x in range(len(bins_0))], [x for x in range(len(bins_1))]],
        names=features,
    )
    eff_df = eff_df.reindex(all_combinations, fill_value=0)
    eff_df = eff_df.subtract(fJ0, level=0).subtract(fJ1, level=1)

    # subtract the total average of a 2D moving average of size 2 (4 cells)
    idx = pd.IndexSlice
    eff_df = (
        eff_df
        - (
            sizes_df
            * (
                eff_df.loc[
                    idx[0 : len(bin_codes_unique_0) - 1],
                    idx[0 : len(bin_codes_unique_1) - 1],
                    :,
                ].values
                + eff_df.loc[
                    idx[1 : len(bin_codes_unique_0)],
                    idx[1 : len(bin_codes_unique_1)],
                    :,
                ].values
                + eff_df.loc[
                    idx[0 : len(bin_codes_unique_0) - 1],
                    idx[1 : len(bin_codes_unique_1)],
                    :,
                ].values
                + eff_df.loc[
                    idx[1 : len(bin_codes_unique_0)],
                    idx[0 : len(bin_codes_unique_1) - 1],
                    :,
                ].values
            )
            / 4
        ).sum()
        / sizes_df.sum()
    )

    # renaming and preparing final output
    eff_df = eff_df.reset_index(name="eff")
    eff_df[features[0]] = bins_0[eff_df[features[0]].values]
    eff_df[features[1]] = bins_1[eff_df[features[1]].values]
    eff_grid = eff_df.pivot_table(columns=features[1], values="eff", index=features[0])

    return eff_grid
