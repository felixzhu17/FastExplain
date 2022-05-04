def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def fill_list(list, length, fill_value=None):
    list += [fill_value] * (length - len(list))
    return list


def drop_duplicate(ls):
    return list(set(ls))


def clean_text(text):
    return str(text).replace("_", " ").title()


def clean_dict_text(d):
    return {k: clean_text(v) for k, v in d.items()}


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


def check_unequal_list(lists):
    return len({len(i) for i in lists}) != 1


def check_numeric(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def check_all_numeric(x):
    return all(check_numeric(i) for i in x)


def try_convert_numeric(x):
    try:
        return [float(i) for i in x]
    except:
        return x
