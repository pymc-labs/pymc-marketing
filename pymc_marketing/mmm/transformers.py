import pytensor.tensor as pt

from pymc_marketing.mmm.utils import params_broadcast_shapes


def batched_convolution(x, w, axis: int = 0):
    """Apply a 1D convolution in a vectorized way across multiple batch dimensions.

    Parameters
    ----------
    x :
        The array to convolve.
    w :
        The weight of the convolution. The last axis of ``w`` determines the number of steps
        to use in the convolution.
    axis : int
        The axis of ``x`` along witch to apply the convolution

    Returns
    -------
    y :
        The result of convolving ``x`` with ``w`` along the desired axis. The shape of the
        result will match the shape of ``x`` up to broadcasting with ``w``. The convolved
        axis will show the results of left padding zeros to ``x`` while applying the
        convolutions.
    """
    orig_ndim = x.ndim
    axis = axis if axis >= 0 else orig_ndim + axis
    w = pt.as_tensor(w)
    x = pt.moveaxis(x, axis, -1)
    try:
        l_max = w.shape[-1].eval()
    except Exception:
        l_max = None
    x_shape, w_shape = params_broadcast_shapes([x.shape, w.shape], [1, 1])
    x = pt.broadcast_to(x, x_shape)
    w = pt.broadcast_to(w, w_shape)
    x_time = x.shape[-1]
    shape = (*x.shape, w.shape[-1])
    padded_x = pt.zeros(shape, dtype=x.dtype)
    if l_max is not None:
        for i in range(l_max):
            padded_x = pt.set_subtensor(
                padded_x[..., i:x_time, i], x[..., : x_time - i]
            )
    else:  # pragma: no cover
        raise NotImplementedError(
            "At the moment, convolving with weight arrays that don't have a concrete shape "
            "at compile time."
        )
    conv = pt.sum(padded_x * w[..., None, :], axis=-1)
    return pt.moveaxis(conv, -1, axis + conv.ndim - orig_ndim)


def geometric_adstock(
    x, alpha: float = 0.0, l_max: int = 12, normalize: bool = False, axis: int = 0
):
    """Geometric adstock transformation.

    Adstock with geometric decay assumes advertising effect peaks at the same
    time period as ad exposure. The cumulative media effect is a weighted average
    of media spend in the current time-period (e.g. week) and previous `l_max` - 1
    periods (e.g. weeks). `l_max` is the maximum duration of carryover effect.


    Parameters
    ----------
    x : tensor
        Input tensor.
    alpha : float, by default 0.0
        Retention rate of ad effect. Must be between 0 and 1.
    l_max : int, by default 12
        Maximum duration of carryover effect.
    normalize : bool, by default False
        Whether to normalize the weights.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling
       with carryover and shape effects." (2017).
    """

    w = pt.power(pt.as_tensor(alpha)[..., None], pt.arange(l_max, dtype=x.dtype))
    w = w / pt.sum(w, axis=-1, keepdims=True) if normalize else w
    return batched_convolution(x, w, axis=axis)


def delayed_adstock(
    x,
    alpha: float = 0.0,
    theta: int = 0,
    l_max: int = 12,
    normalize: bool = False,
    axis: int = 0,
):
    """Delayed adstock transformation.

    This transformation is similar to geometric adstock transformation, but it
    allows for a delayed peak of the effect. The peak is assumed to occur at `theta`.

    Parameters
    ----------
    x : tensor
        Input tensor.
    alpha : float, by default 0.0
        Retention rate of ad effect. Must be between 0 and 1.
    theta : float, by default 0
        Delay of the peak effect. Must be between 0 and `l_max` - 1.
    l_max : int, by default 12
        Maximum duration of carryover effect.
    normalize : bool, by default False
        Whether to normalize the weights.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling
       with carryover and shape effects." (2017).
    """
    w = pt.power(
        pt.as_tensor(alpha)[..., None],
        (pt.arange(l_max, dtype=x.dtype) - pt.as_tensor(theta)[..., None]) ** 2,
    )
    w = w / pt.sum(w, axis=-1, keepdims=True) if normalize else w
    return batched_convolution(x, w, axis=axis)


def logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation.

    Parameters
    ----------
    x : tensor
        Input tensor.
    lam : float, optional, by default 0.5
        Saturation parameter.

    Returns
    -------
    tensor
        Transformed tensor.
    """
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


def tanh_saturation(x, b: float = 0.5, c: float = 0.5):
    """Tanh saturation transformation.

    Parameters
    ----------
    x : tensor
        Input tensor.
    b : float, by default 0.5
        Number of users at saturation. Must be non-negative.
    c : float, by default 0.5
        Initial cost per user. Must be non-zero.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    See https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/ # noqa: E501
    """
    return b * pt.tanh(x / (b * c))
