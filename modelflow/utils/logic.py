def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def fill_list(list, length, fill_value=None):
    list += [fill_value] * (length - len(list))
    return list


def drop_duplicate(ls):
    return list(set(ls))


def clean_text(text):
    return text.replace("_", " ").title()


def bin_columns(bins, dp=2, percentage=False, condense_last=True):
    formatting = "%" if percentage else "f"

    binned_columns = [
        f"{lower:,.{dp}{formatting}} - {upper:,.{dp}{formatting}}"
        for lower, upper in zip(bins[:-1], bins[1:])
    ]

    if condense_last:
        binned_columns[-1] = f"{float(bins[-2]):,.{dp}{formatting}}+"

    return binned_columns


def bin_intervals(bins, dp=2, percentage=False, condense_last=True):
    formatting = "%" if percentage else "f"
    binned_intervals = [
        f"{i.left:,.{dp}{formatting}} - {i.right:,.{dp}{formatting}}" for i in bins
    ]
    if condense_last:
        binned_intervals[-1] = f"{bins[-1].left:,.{dp}{formatting}}+"
    return binned_intervals
