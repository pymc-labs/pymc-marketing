import aesara.tensor as at


def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12):
    """Geometric adstock transformation.

    Parameters
    ----------
    x : tensor
        Input tensor
    alpha : float, by default 0.0
         Retention rate of ad effect
    l_max : int, by default 12
        Maximum duration of carryover effect

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling
    with carryover and shape effects." (2017).
    """
    cycles = [at.concatenate([at.zeros(i), x[: x.shape[0] - i]]) for i in range(l_max)]
    x_cycle = at.stack(cycles)
    w = at.as_tensor_variable([at.power(alpha, i) for i in range(l_max)])
    return at.dot(w, x_cycle)


def logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation.

    Parameters
    ----------
    x : tensor
        Input tensor
    lam : float, optional, by default 0.5
        Saturation parameter

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    See reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget
    in https://www.pymc-labs.io/blog-posts/
    """
    return (1 - at.exp(-lam * x)) / (1 + at.exp(-lam * x))


def tanh_saturation(x, b: float = 0.5, c: float = 0.5):
    """Tanh saturation transformation.

    Parameters
    ----------
    x : tensor
        Input tensor.
    b : float, by default 0.5
        Number of users at saturation.
    c : float, by default 0.5
        Initial cost per user.

    Returns
    -------
    tensor
        Transformed tensor.
    """
    return b * at.tanh(x / (b * c))
