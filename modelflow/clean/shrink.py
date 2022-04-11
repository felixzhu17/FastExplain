import numpy as np


def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int16, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint16, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }
    if obj2cat:
        typemap[
            "object"
        ] = "category"  # User wants to categorify dtype('Object'), which may not always save space
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()),
                None,
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes


def df_shrink(df, skip=[], obj2cat=False, int2uint=False):
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    return df.astype(dt)
