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

    binned_intervals = [
        f"{format_number(i.left, dp, percentage)} - {format_number(i.right, dp, percentage)}"
        if i not in ["NaN"]
        else "NaN"
        for i in bins
    ]

    if condense_last:
        replace_index = -2 if "NaN" in bins else -1
        binned_intervals[
            replace_index
        ] = f"{format_number(bins[replace_index].left, dp, percentage)}+"

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


def check_list_type(x):
    return isinstance(x, (list, tuple))


def doc_setter(origin):
    def wrapper(func):
        func.__doc__ = origin.__doc__
        return func

    return wrapper


def format_number(x, dp, percentage):
    formatting = "%" if percentage else "f"
    return f"{x:,.{dp}{formatting}}"
