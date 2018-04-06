def subdict(dct, keys, mapping=None):
    if mapping is None:
        mapping = {}
    return {mapping.get(k, k): v for k, v in dct.items() if k in keys}


def trim(dct, keys):
    for key in keys:
        if key in dct:
            del dct[key]
    return dct


def extract(dct, keys, mapping=None):
    out = subdict(dct, keys, mapping)
    trim(dct, keys)
    return out


def dict_union(x, y):
    z = y.copy()
    z.update(x)
    return z


def transpose(d):
    keys, vals = zip(*d.items())
    return [dict(zip(keys, v)) for v in zip(*vals)]


def get_recursively(dct, key, *keys, default=None):
    value = dct
    for k in (key,) + keys:
        value = value.get(k, {})
    return value or default


def dict_assert_unique(iterable, assert_unique=True,
                       dict_class=dict):
    if not assert_unique:
        return dict_class(iterable)
    d = dict_class()
    duplicates = set()
    for k, v in iterable:
        if k in d:
            duplicates.add(k)
        else:
            d[k] = v
    if duplicates:
        raise ValueError("Duplicate keys: {}".format(", ".join(duplicates)))
    return d
