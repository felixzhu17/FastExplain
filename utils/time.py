import pandas as pd


class TimeUtils:
    def subtract_time(self, day, delta, unit="d"):
        before_day = pd.to_datetime(day) - pd.Timedelta(delta, unit)
        return before_day.strftime("%Y-%m-%d")

    def chunk_dates(self, start_date, end_date, freq, return_range=False):
        dates = [
            i.strftime("%Y-%m-%d")
            for i in pd.date_range(start=start_date, end=end_date, freq=freq)
        ]
        if return_range:
            for start, end in zip(dates[:-1], dates[1:]):
                yield start, end
        else:
            for date in dates:
                yield date

    def bin_columns(
        self, bins, format_numbers=False, dp=2, percentage=False, condense_last=True
    ):
        if format_numbers:
            if percentage:
                binned_columns = [
                    f"{lower:,.{dp}%} - {upper:,.{dp}%}"
                    for lower, upper in zip(bins[:-1], bins[1:])
                ]

                if condense_last:
                    binned_columns[-1] = f"{float(bins[-2]):,.{dp}%}+"

            else:
                binned_columns = [
                    f"{lower:,.{dp}f} - {upper:,.{dp}f}"
                    for lower, upper in zip(bins[:-1], bins[1:])
                ]
                if condense_last:
                    binned_columns[-1] = f"{float(bins[-2]):,.{dp}f}+"

        else:
            binned_columns = [
                f"{lower} - {upper}" for lower, upper in zip(bins[:-1], bins[1:])
            ]
            if condense_last:
                binned_columns[-1] = f"{float(bins[-2])}+"

        return binned_columns

    def bin_intervals(
        self, bins, format_numbers=False, dp=2, percentage=False, condense_last=True
    ):
        if format_numbers:
            if percentage:
                binned_intervals = [
                    f"{i.left:,.{dp}%} - {i.right:,.{dp}%}" for i in bins
                ]
                if condense_last:
                    binned_intervals[-1] = f"{bins[-1].left:,.{dp}%}+"

            else:
                binned_intervals = [
                    f"{i.left:,.{dp}f} - {i.right:,.{dp}f}" for i in bins
                ]
                if condense_last:
                    binned_intervals[-1] = f"{bins[-1].left:,.{dp}f}+"

        else:
            binned_intervals = [f"{i.left} - {i.right}" for i in bins]
            if condense_last:
                binned_intervals[-1] = f"{bins[-1].left}+"

        return binned_intervals
