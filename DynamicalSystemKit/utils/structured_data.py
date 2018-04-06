import collections
import numpy

from . import dct
from .. import code_generator, numerics


def slc_range(slc):
    return range(slc_start(slc), slc_stop(slc))


def slc_len(slc):
    return slc.stop - slc.start if isinstance(slc, slice) else 1


def slc_stop(slc):
    return slc.stop if isinstance(slc, slice) else slc + 1


def slc_start(slc):
    return slc.start if isinstance(slc, slice) else slc


def slc_join(slc_a, slc_b):
    return slice(slc_start(slc_a), slc_stop(slc_b))


def slc_empty(slc):
    start = slc.start
    stop = slc.stop
    if start is None:
        return stop is not None and stop == 0
    if stop is None:
        return False
    step = slc.step or 1
    return (start >= 0 and stop >= 0 and (stop - start) * step <= 0) \
        or (start < 0 and stop < 0 and (stop - start) * step <= 0)


def _slc_compose_int(outer, i, allow_none=False):
    lim = {True: 0, False: -1}[allow_none]
    if not isinstance(outer, slice):
        raise IndexError("Index out of range")
    step = outer.step or 1
    if i >= 0:
        pos = outer.start or 0
        out = pos + i * step
        if pos < 0:
            if out > lim:
                raise IndexError("Index out of range")
            if out == 0:
                return None
        elif outer.stop is not None and outer.stop > 0 and (out - outer.stop) * step > 0:
            out = outer.stop
        return out
    pos = outer.stop or 0
    if step != 1 and step != -1:
        if outer.start is None or outer.stop is None:
            raise ValueError("Undefined behaviour")
        pos -= ((outer.stop - outer.start) % step)

    out = pos + i * step
    if pos >= 0:
        if out < 0:
            raise IndexError("Index out of range")
        if outer.start is not None and outer.start > 0 and (out - outer.start) * step < 0:
            out = outer.start
    return out

# TODO: support for negative steps


def slc_compose(outer, inner):
    if isinstance(inner, int):
        return _slc_compose_int(outer, inner)
    if slc_empty(outer):
        return slice(0, 0, None)
    if inner.start is None:
        start = outer.start
    else:
        start = _slc_compose_int(outer, inner.start)

    if inner.stop is None:
        stop = outer.stop
    else:
        stop = _slc_compose_int(outer, inner.stop, allow_none=True)
    if inner.step is None:
        step = outer.step
    else:
        step = inner.step * (outer.step or 1)
    return slice(start, stop, step)


def slicify(slc):
    if not isinstance(slc, slice):
        return slice(slc, slc + 1)

    return slc


def slc_complement(slc, N):
    mask = numpy.ones(N, dtype=bool)
    mask[slc] = False
    try:
        switch, = numpy.flatnonzero(numpy.diff(mask))
        if mask[0]:
            return slice(None, switch+1)
        return slice(switch+1, None)
    except ValueError:
        return mask


def flat_slice_l(n, l):
    if l == 1:
        return n
    return slice_l(n, l)


def slice_l(n, l):
    return slice(n, n + l)


def slc_enumerator(slice_type=slice_l):
    _n = 0

    def _l2s(l):
        nonlocal _n
        slc = slice_type(_n, l)
        _n += l
        return slc
    return _l2s


def xlen_to_slc(lengths,
                flatten_slice=False):
    l2s = slc_enumerator(flat_slice_l if flatten_slice else slice_l)
    for l in lengths:
        yield l2s(l)


def tuple_len_to_slc(tuples, **kwargs):
    return tuple(xtuple_len_to_slc(tuples, **kwargs))


def xtuple_len_to_slc(tuples,
                      flatten_slice=False):
    l2s = slc_enumerator(flat_slice_l if flatten_slice else slice_l)
    for *keys, l in tuples:
        yield keys + [l2s(l)]


def len_to_slc_dict(named_lengths, flatten_slice=False, **kwargs):
    return dct.dict_assert_unique(xtuple_len_to_slc(named_lengths, flatten_slice=flatten_slice),
                                  **kwargs)


def slc_to_view(slc, data, axis=-1):
    return data[numerics.idx(slc, axis)]


def xlen_to_view(iterable, data, axis=-1, **kwargs):
    return (slc_to_view(slc, data, axis=axis) for slc in xlen_to_slc(iterable, **kwargs))


def slc_dict_to_view(dct_, data, **kwargs):
    return type(dct_)((key, slc_to_view(slc, data, **kwargs)) for key, slc in dct_.items())


def xtuple_len_to_view(named_lengths, data, axis=-1, **kwargs):
    for *keys, slc in xtuple_len_to_slc(named_lengths, **kwargs):
        yield keys + [slc_to_view(slc, data, axis=axis)]


def len_to_view_dict(data, named_lengths, assert_unique=True, dict_class=dict, **kwargs):
    return dct.dict_assert_unique(xtuple_len_to_view(named_lengths, data, **kwargs),
                                  assert_unique=assert_unique, dict_class=dict_class)


def view_dict_interpolate(col_dict, x, x_col=None, method=numerics.interp_linear):
    if x_col is None and not isinstance(col_dict, collections.OrderedDict):
        raise ValueError("x_ref not specified while timeseries is not an OrderedDict. "
                         "A random x_ref would be chosen!")
    out = type(col_dict)()
    if x_col is None:
        _capt, x_orig = next(col_dict.items())
    else:
        x_orig = col_dict[x_col]
    for caption, y_orig in col_dict.items():
        if caption == x_col:
            out[caption] = x
        else:
            out[caption] = method(x_orig, y_orig, x)
    return out


def subdivide_named_lengths(named_lengths, sub_fields):
    out = ([], [])
    for n, l in named_lengths:
        out[n in sub_fields].append((n, l))
    return out[True], out[False]


def index_list_from_named_lengths(named_lengths, sub_fields):
    slc_dict = len_to_slc_dict(named_lengths)

    def slcs_to_list(slcs):
        for slc in slcs:
            yield from range(slc.start, slc.stop)
    return numpy.fromiter(slcs_to_list(slc_dict[name] for name in sub_fields), dtype=int)


def slice_mapping(named_lengths_src, named_lengths_dest, name_mapping=None):

    name_mapping = name_mapping or {}
    dest_slices = len_to_slc_dict(named_lengths_dest, flatten_slice=True)

    slice_pairs = _get_slice_pairs(
        named_lengths_src, name_mapping, dest_slices)
    try:
        last_src, last_dest = next(slice_pairs)
    except StopIteration:
        return
    for src, dest in slice_pairs:
        if slc_start(src) == slc_stop(last_src) \
           and slc_start(dest) == slc_stop(last_dest):
            last_src = slc_join(last_src, src)
            last_dest = slc_join(last_dest, dest)
            continue
        yield (src, dest)
    yield (last_src, last_dest)


def _get_slice_pairs(named_lengths_src, name_mapping, dest_slices):
    for name, slc_src in xtuple_len_to_slc(named_lengths_src, flatten_slice=True):
        name_dest = name_mapping.get(name, name)
        slc_dest = dest_slices.get(name_dest)
        if slc_dest is None:
            continue

        if slc_len(slc_src) != slc_len(slc_dest):
            raise ValueError(
                "Length mismatch for column {} ({} != {})!".format(
                    name, slc_len(slc_src), slc_len(slc_dest)))
        yield (slc_src, slc_dest)


def data_mapping_n__n(headers_src, headers_dest, axis=-1, **kwargs):
    args = tuple("x" + str(i) for i in range(len(headers_src)))
    code_lines = []
    if len(headers_dest) == 1:
        outs = ("out",)
    else:
        outs = tuple("out" + str(i) for i in range(len(headers_dest)))
    for named_lengths_dest, out in zip(headers_dest, outs):
        n = sum(l for _, l in named_lengths_dest)
        code_lines.append(
            "if {out} is None: {out} = numpy.empty({shp})".format(
                out=out,
                shp=("x0.shape[:{}] +".format(axis) if axis is not 0 else "")
                + "({},)".format(n)
                + (" + x0.shape[{}:]".format(axis + 1)
                   if axis is not -1 else "")
            ))
    for named_lengths_dest, out in zip(headers_dest, outs):
        n_dest = sum(l for _, l in named_lengths_dest)
        for named_lengths_src, arg in zip(headers_src, args):
            n_src = sum(l for _, l in named_lengths_src)
            slc_mapping = slice_mapping(
                named_lengths_src, named_lengths_dest, **kwargs)

            def slc_to_str(slc, axis, n):
                if slc_len(slc) == n:
                    return ":"
                if axis >= 0:
                    return ":," * axis + str(slicify(slc)) + ", ..."
                return "...," + str(slicify(slc)) + ",:" * (-axis - 1)
            for slc_src, slc_dest in slc_mapping:
                code_lines.append("{}[{}] = {}[{}]".format(
                    out,
                    slc_to_str(slc_dest, axis, n_dest),
                    arg,
                    slc_to_str(slc_src, axis, n_src)))

    code_lines.append("return " + ", ".join(outs))
    return code_generator.Function.build(args=args,
                                         kwargs={out: "None" for out in outs},
                                         code_lines=code_lines,
                                         environment={"numpy": numpy})


def data_mapping(named_lengths_src, named_lengths_dest, **kwargs):
    return data_mapping_n__n((named_lengths_src,), (named_lengths_dest,), **kwargs)


def csv_header(named_lengths, delimiter=";"):
    return delimiter.join(name + delimiter * (l - 1) for name, l in named_lengths)
