import numpy
import scipy.interpolate
import scipy.integrate


def interp_linear(x0, y0, x, out=None, axis=0):
    interp = scipy.interpolate.interp1d(x0, y0, axis=axis, copy=False,
                                        assume_sorted=True,
                                        bounds_error=False)(x)
    if out is not None:
        out[()] = interp
    return interp


def interp_step(x0, y0, x, out=None):
    step_edges = numpy.searchsorted(x0, x, side='right') - 1
    interp = y0[step_edges]
    if out is not None:
        out[()] = interp
    return interp


def idx(i, axis=0):
    if axis == 0:
        return i
    if axis < 0:
        return (..., i) + (slice(None),) * (1 + axis)
    return (slice(None),) * axis + (i,)


def rolling_mean(x, w, out=None):
    n = len(x)
    if out is None:
        out = numpy.empty(n - w + 1)
    out[0] = numpy.sum(x[:w])
    out[1:] = x[w:] - x[:-w]
    out[1:] = out[0] + numpy.cumsum(out[1:])
    out /= w
    return out


def rolling_mean_fill_start(x, w, out=None):
    n = len(x)
    if out is None:
        out = numpy.empty(n)
    rolling_mean(x, w, out=out[w - 1:])
    out[:w - 1] = out[w - 1]
    return out


def rolling_mean_fill_start_and_end(x, w, out=None):
    n = len(x)
    if out is None:
        out = numpy.empty(n)
    w1 = w // 2
    w2 = w - 1 - w1
    rolling_mean(x, w, out=out[w1:-w2])
    out[:w1] = out[w1]
    out[-w2:] = out[-w2 - 1]
    return out


def lsq_normalize(A, b, lsq_ext=None, eps_lsq=0., x_norm=None, **kwargs):
    """
    Optimize for x in the equation A x + eps = b, such that in
    A_nrm[i] x_nrm[i] + eps = b_nrm, eps has least squares, where
    A_nrm[i] = (A[i] - mean(A[i]))/std(A[i]),
    b_nrm = (b - mean(b))/std(b).
    x_nrm[i] = std(x[i])/std(b) * x[i]
    The return value is x.
    Warning: A, b are modified!
    """
    if lsq_ext is None:
        lsq_ext = (lambda X, y: numpy.linalg.lstsq(X, y, rcond=-1)[0])
    # print("Matrix rank:", numpy.linalg.matrix_rank(A), "/", min(A.shape))
    A_mean = numpy.mean(A, axis=-2)
    A -= A_mean
    A_std = numpy.std(A, axis=-2)
    A_std[A_std == 0.] = 1.
    A_std_inv = 1. / A_std
    A *= A_std_inv
    b_mean = numpy.mean(b, axis=-1)
    b_std = numpy.array(numpy.std(b, axis=-1), copy=False)
    b_std[b_std == 0.] = 1.
    b -= b_mean
    b /= b_std
    x_rescale = A_std_inv * b_std
    # print("Matrix rank (normalized):",
    #       numpy.linalg.matrix_rank(A), "/", min(A.shape))
    if eps_lsq != 0.:
        kwargs["eps"] = numpy.diag(x_rescale)
    x = lsq_ext(A, b, **kwargs)
    if x_norm is not None:
        x_norm[:] = x
    x *= x_rescale
    eps_mean = b_mean - numpy.sum(A_mean * x, axis=-1)

    return x, eps_mean


def lsq(A, b, eps=0., **kwargs):
    if eps == 0.:
        eps = None
    else:
        eps = eps * numpy.eye(A.shape[-1])
    return lsq_eps(A, b, eps=eps, **kwargs)


def lsq_eps(A, b, fill_singular=float("NaN"), eps=None):
    """
    Vectorized version of numpy.linalg.lstsq including
    regularization with a matrix eps
    """
    A_T = numpy.swapaxes(A, -2, -1)
    AA = numpy.matmul(A_T, A)
    if eps is not None:
        AA += eps
    V = gram_schmidt(AA)
    mask_x = numpy.isclose(0., numpy.sum(numpy.abs(V), axis=-2))
    if mask_x.any():
        # cancel singular columns and rows:
        mask_AA = numpy.logical_or(mask_x[..., None, :], mask_x[..., :, None])
        AA[mask_AA] = 0.
        # Set singular diagonal elements to 1 => matrix is regular:
        mask_AA = numpy.logical_and(mask_x[..., None, :], mask_x[..., :, None])
        AA[mask_AA] = 1.
    Ab = numpy.sum(b[..., None] * A, axis=-2)
    x = numpy.linalg.solve(AA, Ab)
    numpy.atleast_2d(x)[mask_x] = fill_singular
    return x


def gram_schmidt(A):
    A = atleast_3d_l(A.copy())
    n = A.shape[-1]
    for i in range(n - 1):
        l = numpy.atleast_1d(numpy.sum(A[..., i] ** 2, axis=-1))
        mask = (l != 0)
        A[mask, :, i + 1:] -= (1. / l[mask, None, None]
                               * numpy.sum(A[mask, :, i + 1:]
                                           * A[mask, :, i, None], axis=-2)[..., None, :]
                               * A[mask, :, i, None])
    return A


def atleast_3d_l(x):
    n = len(numpy.shape(x))
    if n >= 3:
        return x
    if n == 2:
        return x[None, :, :]
    if n == 1:
        return x[None, None, :]
    if n == 0:
        return numpy.array([[[x]]])
    return x


def broadcast_shape(shp_a, shp_b):
    if len(shp_b) > len(shp_a):
        return broadcast_shape(shp_b, shp_a)
    dn = len(shp_a) - len(shp_b)
    shp_head = shp_a[:dn]

    def bc_indices(i, j):
        if i == 1:
            return j
        if i != j and j != 1:
            raise ValueError(
                "shapes {}, {} not broadcastable!".format(shp_a, shp_b))
        return i

    return shp_head + tuple(bc_indices(i, j) for i, j in zip(shp_a[dn:], shp_b))


def broadcast_shape_multi(*shp):
    out_shp = ()
    for s in shp:
        out_shp = broadcast_shape(s, out_shp)
    return out_shp


def concatenate(*dat, axis=0, out=None):
    if out is None:
        N = sum(numpy.atleast_1d(x).shape[axis] for x in dat)
        shp = dat[0].shape[:axis] + (N,) + dat[0].shape[axis + 1:]
        out = numpy.empty(shp)
    n = 0
    for x in dat:
        m = n + numpy.atleast_1d(x).shape[axis]
        out[idx(slice(n, m), axis)] = x
        n = m
    return out


def interval_slice(t, t0, t1):
    n0 = None if t0 is None else numpy.searchsorted(t, t0)
    n1 = None if t1 is None else numpy.searchsorted(t, t1, side="right")
    return slice(n0, n1)


def crop(t0, t1, X, axis=0, t=None):
    if axis < 0:
        n0 = X.ndim + axis
        n1 = -axis - 1
    else:
        n0 = axis
        n1 = X.ndim - axis - 1
    I = (0,) * n0 + (slice(None),) + (0,) * n1
    if t is None:
        t = X[I]
    slc = interval_slice(t, t0, t1)
    return X[idx(slc, axis=axis)]


def crop_t(t0, t1, t, *X):
    slc = interval_slice(t, t0, t1)
    t_slc = t[slc]
    if not X:
        return t_slc
    return (t_slc,) + tuple(x[slc] for x in X)


def cumtrapz(x, t, axis=0, y0=0., out=None):
    _out = y0 + scipy.integrate.cumtrapz(x, x=t, initial=0., axis=axis)
    if out is not None:
        out[:] = _out
    return _out


def cumrect(x, t, axis=0, y0=0., out=None):
    if out is None:
        out = numpy.empty_like(x)

    out[idx(0, axis=axis)] = y0
    dn = -1 - axis
    if dn < 0:
        dn += x.ndim
    t = t.reshape(t.shape + (1,) * dn)
    out[idx(slice(1, None), axis=axis)] \
        = x[idx(slice(None, -1), axis=axis)] * (t[1:] - t[:-1])
    numpy.cumsum(out, axis=axis, out=out)
    return out


def trapz(t0, x0, t, initial=0.):
    """
    Return the integral of the polygonal chain
    defined by t0, x0 at the points t
    """
    t_full, idx_full = numpy.unique(
        numpy.concatenate((t, t0)), return_index=True)
    mask_t = (idx_full < t.shape[0])
    x_full = numpy.interp(t_full, t0, x0)
    X = scipy.integrate.cumtrapz(x_full, t_full, initial=initial)
    return X[mask_t]


def distribution(t, x, bin_seps, period):
    t0 = bin_seps[0]
    t_period = (t - t0) % period
    bin_seps = bin_seps - t0
    bin_counts = numpy.histogram(t_period, bins=bin_seps,
                                 range=(0., period))[0]
    return numpy.histogram(t_period, bins=bin_seps,
                           weights=x, range=(0., period))[0] / bin_counts


def periods_raster(t, period_borders, period, out=None, out_extra_shape=()):
    """
    period_borders:       ||  |
    period:              ----------
    t:               ** *  * * **** **  *    * *
    raster:          |    ||  |    ||  |    ||
    """
    n_periods = len(period_borders)
    n_min = int(t[0] // period)
    i_min = numpy.searchsorted(period_borders, t[0] % period)
    n_max = int(t[-1] // period)
    i_max = numpy.searchsorted(period_borders, t[-1] % period, side="right")
    n_cells = n_max - n_min

    if n_cells == 0:
        if out is None:
            out = numpy.empty((0,) + out_extra_shape)
        return out
    N = n_cells * n_periods - i_min + i_max
    if out is None:
        out = numpy.empty((N,) + out_extra_shape)
    out_view = out[(slice(None),) + (0,) * len(out.shape[1:])]
    t0 = n_min * period
    slc_middle = slice(n_periods - i_min, n_cells * n_periods - i_min)
    # pylint: disable=invalid-slice-index
    out_view[:slc_middle.start] = t0 + period_borders[i_min:]
    middle_view = out_view[slc_middle]
    middle_view[:] = numpy.tile(period_borders, n_cells - 1)
    middle_view.reshape(n_cells - 1, n_periods)[:] \
        += numpy.arange(n_min + 1, n_max)[:, None] * period
    out_view[slc_middle.stop:] = n_max * period + period_borders[:i_max]
    return out


def bin_sums(dist, bin_seps, out=None):
    out = out or numpy.empty(bin_seps.shape)
    dist_sum = numpy.empty(dist.shape[:-1] + (dist.shape[-1] + 1,))
    numpy.cumsum(dist, out=dist_sum[..., 1:], axis=-1)
    dist_sum[..., 0] = 0
    out[..., :-1] = numpy.diff(dist_sum[bin_seps], axis=-1)
    out[..., -1] = dist_sum[-1] - dist_sum[bin_seps[..., -1]] \
        + dist_sum[bin_seps[..., 0]]
    return out


def bin_means(dist, bin_seps, out=None):
    bin_lengths = numpy.empty(bin_seps.shape, dtype=int)
    bin_lengths[..., :-1] = numpy.diff(bin_seps, axis=-1)
    n = dist.shape[-1]
    bin_lengths[..., -1] = bin_seps[..., 0] + n - bin_seps[..., -1]

    out = bin_sums(dist, bin_seps, out=out)
    out /= bin_lengths
    return out


def bin_mean_deviation(dist, bin_seps):
    means = bin_means(dist, bin_seps)
    n = dist.shape[-1]
    deviations = dist - staircase(bin_seps, means, n)
    numpy.absolute(deviations, out=deviations)
    return numpy.sum(deviations, axis=-1)


def staircase(i0, y0, n):
    """
    Generates a staircase array over the indices 0,...,n-1 with values y0[i]
    at the index positions i0[i],...,i0[i+1]. It is cyclic, i.e. the level
    starting at y0[-1] is continued at 0 and ends at i0[0] (if i0[0] > 0).
    """
    sh = i0.shape[:-1]
    idx_ = numpy.zeros(sh + (n,), dtype=int)
    if sh:
        di = (numpy.arange(sh[-1], dtype=int) * n)[:, None]
        idx_0 = (numpy.arange(sh[-1], dtype=int)[:, None], )
    else:
        di = 0
        idx_0 = ()

    numpy.put(idx_, (i0 + di).flatten(), 1)
    idx_ = numpy.cumsum(idx_, axis=-1) - 1
    return y0[idx_0 + (idx_,)]


def slc_overlap_l(x, y):
    if not y:
        return slice(0, 0)
    i0 = numpy.searchsorted(x, y[0])
    i1 = numpy.searchsorted(x, y[-1], side="right")
    return slice(i0, i1)


def slc_overlap(x, y, assert_match=True):
    slc_x = slc_overlap_l(x, y)
    slc_y = slc_overlap_l(y, x)
    if assert_match and numpy.any(x[slc_x] != y[slc_y]):
        raise ValueError("Input arrays do not match along overlap!")
    return slc_x, slc_y


def prm_density_distribution(t, x, y):
    r"""
    Given x(t), y(t) assumed as polyline, compute the density distribution y(x)
    y(x) = d/dx \int_{t_0}^t_1 y(t) \chi_{x(t) < x} dt
    x(t) = x_i + dx_i (t-t_i), t_i <= t <= t_{i+1}
    => y(x) = \sum_i y(t_i + (x-x_i)/dx_i)/|dx_i| \chi_{[x_i, x_{i+1}]}(x)
    """
    dt = (t[1:] - t[:-1])
    X = numpy.column_stack((x, y))
    dx_t, dy_t = (X[1:] - X[:-1]).T / dt
    y_x = y[:-1] / numpy.abs(dx_t)
    dy_x = dy_t / (numpy.abs(dx_t) * dx_t)
    y_x -= x[:-1] * dy_x
    x_ordered, x_lookup = numpy.unique(x, return_inverse=True)
    y_ordered, dy_ordered = numpy.full((2, x_ordered.shape[0] - 1), 0.)
    for _y0, _dy, _i0, _i1 in zip(y_x, dy_x, x_lookup[:-1], x_lookup[1:]):
        if _i1 < _i0:
            _i0, _i1 = (min(_i0, _i1), max(_i0, _i1))
        slc = slice(_i0, _i1)
        y_ordered[slc] += (_y0 + _dy * x_ordered[slc])
        dy_ordered[_i0:_i1] += _dy
    return steps_to_polyline(x_ordered, y_ordered, dy_ordered)


def prm_histogram(t, x, y, dx=1.):
    r"""
    Given x(t), y(t) assumed as polyline, compute the histogram y vs x
    """
    x_ordered = raster(numpy.min(x), numpy.max(x), dx)
    if not numpy.shape(y):
        y = numpy.full_like(x, y)
    Y = bintrapz(y, t)
    y_ordered, _x_ordered = numpy.histogram(x, bins=x_ordered, weights=Y)
    return numpy.column_stack((0.5 * (x_ordered[:-1] + x_ordered[1:]), y_ordered))


def raster(x_min, x_max, dx):
    n_min = int(numpy.floor(x_min / dx))
    n_max = int(numpy.ceil(x_max / dx))
    return numpy.arange(dx * n_min, dx * n_max, dx)


def bintrapz(x, t=None):
    X = numpy.empty_like(x)
    if t is None:
        dt = 1.
    else:
        dt = t[1:] - t[:-1]
    x_m = 0.5 * (x[1:] + x[:-1])
    X[:-1] = 0.25 * (x_m + x[:-1]) * dt
    X[-1] = 0.
    X[1:] += 0.25 * (x_m + x[1:]) * dt
    return X


def steps_to_polyline(x, y, dy, out=None):
    n_x = x.shape[0]
    if out is None:
        out = numpy.empty((n_x * 2 - 2, 2))
    ddy = (x[1:] - x[:-1]) * dy
    out_rshp = out.reshape((n_x - 1, 2, 2))
    out_rshp[:, 0, 0] = x[:-1]
    out_rshp[:, 0, 1] = y
    out_rshp[:, 1, 0] = x[1:]
    out_rshp[:, 1, 1] = y + ddy
    # Remove duplicate points:
    mask = numpy.empty(out.shape[0], dtype=bool)
    mask[0] = True
    mask[1:] = numpy.any(out[1:] - out[:-1], axis=-1)
    return out[mask]
