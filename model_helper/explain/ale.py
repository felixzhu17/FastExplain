import pandas as pd
import plotly.express as px
from ...utils import Utils
from ..PyALE import ale
from ..PyALE._src.ALE_2D import aleplot_2D_continuous


class Ale(Utils):
    def ale_summary(self, m, xs, col, model_names=None, *args, **kwargs):
        if isinstance(m, (list, tuple)):
            model_names = (
                model_names if model_names else [f"Model {i}" for i in range(len(m))]
            )
            ales = []
            for count, ale_info in enumerate(zip(m, xs)):
                model, x_values = ale_info
                if count == len(m) - 1:
                    ales.append(
                        self._clean_ale(model, x_values, col, *args, **kwargs)[
                            ["eff", "size"]
                        ]
                    )
                else:
                    ales.append(
                        self._clean_ale(model, x_values, col, *args, **kwargs)[["eff"]]
                    )

            output = self.merge_multi_df(ales, left_index=True, right_index=True)
            output.columns = model_names + ["size"]
            return output
        else:
            return self._clean_ale(m, xs, col, *args, **kwargs)

    def _clean_ale(
        self,
        m,
        xs,
        col,
        normalize=True,
        percentage=False,
        condense_last=True,
        remove_last_bins=None,
        dp=2,
        filter=None,
        *args,
        **kwargs,
    ):
        if filter:
            xs = xs.query(filter)
        df = ale(xs, model=m, feature=[col], plot=False, *args, **kwargs)
        df = df[~df.index.duplicated(keep="last")]
        adjust = -1 * df.iloc[0]["eff"]
        df["eff"] += adjust
        df["lowerCI_95%"] += adjust
        df["upperCI_95%"] += adjust
        if normalize:
            df.index = self.convert_ale_index(
                pd.to_numeric(df.index), dp, percentage, condense_last
            )
        if remove_last_bins:
            df = df.iloc[:-remove_last_bins]
        return df

    def plot_ale(
        self,
        m,
        xs,
        col,
        feature_name=None,
        dep_name=None,
        model_names=None,
        plotsize=None,
        *args,
        **kwargs,
    ):

        feature_name = feature_name if feature_name else self.clean_text(col)

        if isinstance(m, (list, tuple)):
            model_names = (
                model_names if model_names else [f"Model {i}" for i in range(len(m))]
            )
            for count, ale_info in enumerate(
                zip(m, xs, model_names, self.cycle_colours())
            ):
                model, x_values, model_name, color = ale_info
                if count == 0:
                    traces, x, size = self._get_ale_traces(
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
                        self._get_ale_traces(
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
            traces, x, size = self._get_ale_traces(
                m,
                xs,
                col,
                feature_name,
                self.blue,
                return_index_size=True,
                *args,
                **kwargs,
            )

        return self.plot_upper_lower_bound_traces(
            traces,
            x,
            size,
            x_axis_title=feature_name,
            y_axis_title=dep_name,
            plotsize=plotsize,
        )

    def _get_ale_traces(
        self, m, xs, col, model_name, color, return_index_size=True, *args, **kwargs
    ):
        df = self.ale_summary(m, xs, col, *args, **kwargs)
        x = df.index
        y = df["eff"]
        size = df["size"]
        y_lower = df["lowerCI_95%"]
        y_upper = df["upperCI_95%"]
        return self._get_upper_lower_bound_traces(
            x, y, y_lower, y_upper, size, color, model_name, return_index_size
        )

    def plot_multi_ale(self, m, xs, cols, index, plotsize=None, *args, **kwargs):
        pdp = {
            i: self.fill_list(
                list(self.ale_summary(m, xs, i, *args, **kwargs)["eff"]), len(index)
            )
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
        self,
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
        df.index = self.convert_ale_index(df.index, dp, percentage, condense_last)
        df.columns = self.convert_ale_index(df.columns, dp, percentage, condense_last)
        return self._plot_two_way_analysis(
            df, cols, feature_names, plotsize, colorscale
        )

    def convert_ale_index(self, index, dp, percentage, condense_last):
        if percentage:
            return [f"{index[0]:,.{dp}%}"] + self.bin_columns(
                index, True, dp=dp, percentage=percentage, condense_last=condense_last
            )
        else:
            return [f"{index[0]:,.{dp}f}"] + self.bin_columns(
                index, True, dp=dp, percentage=percentage, condense_last=condense_last
            )


class AleClassified(Ale):
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    def ale_summary(self, *args, **kwargs):
        return Ale().ale_summary(self.m, self.xs, *args, **kwargs)

    def plot_ale(self, *args, **kwargs):
        return Ale().plot_ale(self.m, self.xs, *args, **kwargs)

    def plot_multi_ale(self, *args, **kwargs):
        return Ale().plot_multi_ale(self.m, self.xs, *args, **kwargs)

    def plot_2d_ale(self, *args, **kwargs):
        return Ale().plot_2d_ale(self.m, self.xs, *args, **kwargs)
