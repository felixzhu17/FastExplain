from itertools import product
import numpy as np
import inspect


def bool_map_list(ls, mapping):
    return [i for i, j in zip(ls, mapping) if j]


def binary_permutations(ls, len_cutoff=None):
    output = [bool_map_list(ls, i) for i in product([True, False], repeat=4)]
    if len_cutoff:
        output = [i for i in output if len(i) >= len_cutoff]
    return output


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def remove_values_list(ls, values):
    return [i for i in ls if i not in values]


def filter_values_list(ls, values):
    return [i for i in ls if i in values]


def fill_list(list, length, fill_value=None):
    list += [fill_value] * (length - len(list))
    return list


def conditional_mean(x, cutoff):
    if len(x) < cutoff:
        return None
    else:
        return np.mean(x)


def extract_args(func, kwargs):
    args = list(inspect.signature(func).parameters)
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in args}
