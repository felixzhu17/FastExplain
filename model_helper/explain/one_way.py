import pandas as pd
from ...utils import Utils


class OneWay(Utils):
    def get_one_way_analysis(
        self,
        df,
        x_col,
        y_col,
        grid_size=20,
        bins=None,
        format_numbers=True,
        dp=2,
        func=None,
        size_cutoff=0,
        percentage=False,
        condense_last=True,
    ):

        bins = bins if bins else self.get_bins(df[x_col], grid_size)
        filtered_df = df[[x_col, y_col]].copy()
        filtered_df[x_col] = pd.cut(filtered_df[x_col], bins, include_lowest=True)
        func = func if func else lambda x: self.conditional_mean(x, size_cutoff)
        one_way_df = filtered_df.groupby(x_col).agg(
            **{y_col: (y_col, func), "size": (y_col, "count")}
        )
        one_way_df.index = self.bin_intervals(
            one_way_df.index, format_numbers, dp, percentage, condense_last
        )
        return one_way_df

    def plot_one_way_analysis(
        self, df, x_col, y_col, feature_names=None, plotsize=None, *args, **kwargs
    ):
        output = self.get_one_way_analysis(df, x_col, y_col, *args, **kwargs)
        return self._plot_one_way_analysis(
            df=output,
            cols=[x_col, y_col],
            size=output["size"],
            feature_names=feature_names,
            plotsize=plotsize,
        )

    def get_two_way_analysis(
        self,
        df,
        x_cols,
        y_col,
        grid_size=20,
        bins=None,
        format_numbers=True,
        dp=2,
        func=None,
        size_cutoff=0,
        percentage=False,
        condense_last=True,
    ):
        col_1, col_2 = x_cols
        if bins:
            if len(bins) != 2:
                raise ValueError("Need two sets of bins to get two-way analysis")
            bin_1 = bins[0]
            bin_2 = bins[1]
        else:
            bin_1 = self.get_bins(df[col_1], grid_size)
            bin_2 = self.get_bins(df[col_2], grid_size)

        filtered_df = df[x_cols + [y_col]].copy()
        filtered_df[col_1] = pd.cut(filtered_df[col_1], bin_1, include_lowest=True)
        filtered_df[col_2] = pd.cut(filtered_df[col_2], bin_2, include_lowest=True)

        func = func if func else lambda x: self.conditional_mean(x, size_cutoff)

        two_way_df = (
            filtered_df.groupby(x_cols)
            .apply(func)
            .reset_index()
            .pivot(index=col_1, columns=col_2)[y_col]
        )

        two_way_df.index = self.bin_intervals(
            two_way_df.index, format_numbers, dp, percentage, condense_last
        )
        two_way_df.columns = self.bin_intervals(
            two_way_df.columns, format_numbers, dp, percentage, condense_last
        )
        return two_way_df

    def get_two_way_frequency(self, df, x_cols, *args, **kwargs):

        mod_df = df.copy()
        mod_df["dummy_count"] = 1

        output = self.get_two_way_analysis(
            df=mod_df, x_cols=x_cols, y_col="dummy_count", func=sum, *args, **kwargs
        )

        return output / output.sum(axis=0)

    def plot_two_way_frequency(
        self,
        df,
        x_cols,
        feature_names=None,
        plotsize=None,
        colorscale="Blues",
        *args,
        **kwargs,
    ):
        output = self.get_two_way_frequency(df, x_cols, *args, **kwargs)
        return self._plot_two_way_analysis(
            output, x_cols, feature_names, plotsize, colorscale
        )

    def plot_two_way_analysis(
        self,
        df,
        x_cols,
        y_col,
        grid_size=20,
        bins=None,
        format_numbers=True,
        dp=2,
        func=None,
        feature_names=None,
        plotsize=None,
        colorscale="Blues",
        size_cutoff=0,
        percentage=False,
        condense_last=True,
    ):
        two_way_df = self.get_two_way_analysis(
            df=df,
            x_cols=x_cols,
            y_col=y_col,
            grid_size=grid_size,
            bins=bins,
            format_numbers=format_numbers,
            dp=dp,
            func=func,
            size_cutoff=size_cutoff,
            percentage=percentage,
            condense_last=condense_last,
        )
        return self._plot_two_way_analysis(
            two_way_df, x_cols, feature_names, plotsize, colorscale
        )


class OneWayClassified(OneWay):
    def __init__(self, m, xs, df, dep_var):
        self.m = m
        self.xs = xs
        self.df = df
        self.dep_var = dep_var

    def get_one_way_analysis(self, x_col, *args, **kwargs):
        return OneWay().get_one_way_analysis(
            self.df, x_col=x_col, y_col=self.dep_var, *args, **kwargs
        )

    def plot_one_way_analysis(self, x_col, *args, **kwargs):
        return OneWay().plot_one_way_analysis(
            self.df, x_col=x_col, y_col=self.dep_var, *args, **kwargs
        )

    def get_two_way_analysis(self, x_cols, *args, **kwargs):
        return OneWay().get_two_way_analysis(
            self.df, x_cols=x_cols, y_col=self.dep_var, *args, **kwargs
        )

    def plot_two_way_analysis(self, x_cols, *args, **kwargs):
        return OneWay().plot_two_way_analysis(
            self.df, x_cols=x_cols, y_col=self.dep_var, *args, **kwargs
        )

    def get_two_way_frequency(self, *args, **kwargs):
        return OneWay().get_two_way_frequency(self.df, *args, **kwargs)

    def plot_two_way_frequency(self, *args, **kwargs):
        return OneWay().plot_two_way_frequency(self.df, *args, **kwargs)
