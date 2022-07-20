import aesara
import aesara.tensor as at


def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12, normalize: bool = False):
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
    x_cycle, _ = aesara.scan(
        fn=lambda i, _, x: at.concatenate([at.zeros(i), x[: x.shape[0] - i]]),
        sequences=at.arange(l_max),
        outputs_info=at.zeros_like(x),
        non_sequences=x,
    )
    w = at.as_tensor_variable([at.power(alpha, i) for i in range(l_max)])
    w = w / at.sum(w) if normalize else w
    return at.dot(w, x_cycle)


def geometric_adstock_vectorized(x, alpha, l_max: int = 12, normalize: bool = False):
    """Vectorized geometric adstock transformation."""
    x_cycle, _ = aesara.scan(
        fn=lambda i, _, x: at.concatenate(
            tensor_list=[at.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]]
        ),
        sequences=at.arange(l_max),
        outputs_info=at.zeros_like(x),
        non_sequences=x,
    )
    x_cycle = at.transpose(x=x_cycle, axes=[1, 2, 0])
    w = at.as_tensor_variable([at.power(alpha, i) for i in range(l_max)])
    w = at.transpose(w)[None, ...]
    w = w / at.sum(w, axis=2, keepdims=True) if normalize else w
    return at.sum(at.mul(x_cycle, w), axis=2)


def delayed_adstock(
    x, alpha: float = 0.0, theta: int = 0, l_max: int = 12, normalize: bool = False
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
    x_cycle, _ = aesara.scan(
        fn=lambda i, _, x: at.concatenate([at.zeros(i), x[: x.shape[0] - i]]),
        sequences=at.arange(l_max),
        outputs_info=at.zeros_like(x),
        non_sequences=x,
    )
    w = at.as_tensor_variable(
        [at.power(alpha, ((i - theta) ** 2)) for i in range(l_max)]
    )
    w = w / at.sum(w) if normalize else w
    return at.dot(w, x_cycle)


def delayed_adstock_vectorized(
    x, alpha, theta, l_max: int = 12, normalize: bool = False
):
    """Delayed adstock transformation."""
    x_cycle, _ = aesara.scan(
        fn=lambda i, _, x: at.concatenate(
            tensor_list=[at.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]]
        ),
        sequences=at.arange(l_max),
        outputs_info=at.zeros_like(x),
        non_sequences=x,
    )
    x_cycle = at.transpose(x=x_cycle, axes=[1, 2, 0])
    w = at.as_tensor_variable(
        [at.power(alpha, ((i - theta) ** 2)) for i in range(l_max)]
    )
    w = at.transpose(w)[None, ...]
    w = w / at.sum(w, axis=2, keepdims=True) if normalize else w
    return at.sum(at.mul(x_cycle, w), axis=2)


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
    return (1 - at.exp(-lam * x)) / (1 + at.exp(-lam * x))


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
    return b * at.tanh(x / (b * c))
