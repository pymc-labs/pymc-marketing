from enum import Enum
from typing import Union

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.tensor.random.utils import params_broadcast_shapes


class ConvMode(Enum):
    After = "After"
    Before = "Before"
    Overlap = "Overlap"


def batched_convolution(x, w, axis: int = 0, mode: ConvMode = ConvMode.Before):
    R"""Apply a 1D convolution in a vectorized way across multiple batch dimensions.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import batched_convolution, ConvMode
        plt.style.use('arviz-darkgrid')
        spends = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        w = np.array([0.75, 0.25, 0.125, 0.125])
        x = np.arange(-5, 6)
        ax = plt.subplot(111)
        for mode in [ConvMode.Before, ConvMode.Overlap, ConvMode.After]:
            y = batched_convolution(spends, w, mode=mode).eval()
            suffix = "\n(default)" if mode == ConvMode.Before else ""
            plt.plot(x, y, label=f'{mode.value}{suffix}')
        plt.xlabel('time since spend', fontsize=12)
        plt.ylabel('f(time since spend)', fontsize=12)
        plt.title(f"1 spend at time 0 and {w = }", fontsize=14)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    Parameters
    ----------
    x :
        The array to convolve.
    w :
        The weight of the convolution. The last axis of ``w`` determines the number of steps
        to use in the convolution.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries of the input signal, denoted as "x." The default mode is ConvMode.Before.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects, where the effect overlaps with both preceding and succeeding elements.

    Returns
    -------
    y :
        The result of convolving ``x`` with ``w`` along the desired axis. The shape of the
        result will match the shape of ``x`` up to broadcasting with ``w``. The convolved
        axis will show the results of left padding zeros to ``x`` while applying the
        convolutions.
    """
    # We move the axis to the last dimension of the array so that it's easier to
    # reason about parameter broadcasting. We will move the axis back at the end
    orig_ndim = x.ndim
    axis = axis if axis >= 0 else orig_ndim + axis
    w = pt.as_tensor(w)
    x = pt.moveaxis(x, axis, -1)
    l_max = w.type.shape[-1]
    if l_max is None:
        try:
            l_max = w.shape[-1].eval()
        except Exception:
            pass
    # Get the broadcast shapes of x and w but ignoring their last dimension.
    # The last dimension of x is the "time" axis, which doesn't get broadcast
    # The last dimension of w is the number of time steps that go into the convolution
    x_shape, w_shape = params_broadcast_shapes([x.shape, w.shape], [1, 1])

    x = pt.broadcast_to(x, x_shape)
    w = pt.broadcast_to(w, w_shape)
    x_time = x.shape[-1]
    # Make a tensor with x at the different time lags needed for the convolution
    x_shape = x.shape
    # Add the size of the kernel to the time axis
    shape = (*x_shape[:-1], x_shape[-1] + w.shape[-1] - 1, w.shape[-1])
    padded_x = pt.zeros(shape, dtype=x.dtype)

    if l_max is None:  # pragma: no cover
        raise NotImplementedError(
            "At the moment, convolving with weight arrays that don't have a concrete shape "
            "at compile time is not supported."
        )
    # The window is the slice of the padded array that corresponds to the original x
    if l_max <= 1:
        window = slice(None)
    elif mode == ConvMode.After:
        window = slice(l_max - 1, None)
    elif mode == ConvMode.Before:
        window = slice(None, -l_max + 1)
    elif mode == ConvMode.Overlap:
        # Handle even and odd l_max differently if l_max is odd then we can split evenly otherwise we drop from the end
        window = slice((l_max // 2) - (1 if l_max % 2 == 0 else 0), -(l_max // 2))
    else:
        raise ValueError(f"Wrong Mode: {mode}, expected of ConvMode")

    for i in range(l_max):
        padded_x = pt.set_subtensor(padded_x[..., i : x_time + i, i], x)

    padded_x = padded_x[..., window, :]

    # The convolution is treated as an element-wise product, that then gets reduced
    # along the dimension that represents the convolution time lags
    conv = pt.sum(padded_x * w[..., None, :], axis=-1)
    # Move the "time" axis back to where it was in the original x array
    return pt.moveaxis(conv, -1, axis + conv.ndim - orig_ndim)


def geometric_adstock(
    x, alpha: float = 0.0, l_max: int = 12, normalize: bool = False, axis: int = 0
):
    R"""Geometric adstock transformation.

    Adstock with geometric decay assumes advertising effect peaks at the same
    time period as ad exposure. The cumulative media effect is a weighted average
    of media spend in the current time-period (e.g. week) and previous `l_max` - 1
    periods (e.g. weeks). `l_max` is the maximum duration of carryover effect.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import geometric_adstock
        plt.style.use('arviz-darkgrid')
        l_max = 12
        params = [
            (0.01, False),
            (0.5, False),
            (0.9, False),
            (0.5, True),
            (0.9, True),
        ]
        spend = np.zeros(15)
        spend[0] = 1
        ax = plt.subplot(111)
        x = np.arange(len(spend))
        for a, normalize in params:
            y = geometric_adstock(spend, alpha=a, l_max=l_max, normalize=normalize).eval()
            plt.plot(x, y, label=f'alpha = {a}\nnormalize = {normalize}')
        plt.xlabel('time since spend', fontsize=12)
        plt.title(f'Geometric Adstock with l_max = {l_max}', fontsize=14)
        plt.ylabel('f(time since spend)', fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

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
    R"""Delayed adstock transformation.

    This transformation is similar to geometric adstock transformation, but it
    allows for a delayed peak of the effect. The peak is assumed to occur at `theta`.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import delayed_adstock
        plt.style.use('arviz-darkgrid')
        params = [
            (0.25, 0, False),
            (0.25, 5, False),
            (0.75, 5, False),
            (0.75, 5, True)
        ]
        spend = np.zeros(15)
        spend[0] = 1
        x = np.arange(len(spend))
        ax = plt.subplot(111)
        for a, t, normalize in params:
            y = delayed_adstock(spend, alpha=a, theta=t, normalize=normalize).eval()
            plt.plot(x, y, label=f'alpha = {a}\ntheta = {t}\nnormalize = {normalize}')
        plt.xlabel('time since spend', fontsize=12)
        plt.ylabel('f(time since spend)', fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

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


def logistic_saturation(x, lam: Union[npt.NDArray[np.float_], float] = 0.5):
    """Logistic saturation transformation.

    .. math::
        f(x) = \\frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import logistic_saturation
        plt.style.use('arviz-darkgrid')
        lam = np.array([0.25, 0.5, 1, 2, 4])
        x = np.linspace(0, 5, 100)
        ax = plt.subplot(111)
        for l in lam:
            y = logistic_saturation(x, lam=l).eval()
            plt.plot(x, y, label=f'lam = {l}')
        plt.xlabel('spend', fontsize=12)
        plt.ylabel('f(spend)', fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    Parameters
    ----------
    x : tensor
        Input tensor.
    lam : float or array-like, optional, by default 0.5
        Saturation parameter.

    Returns
    -------
    tensor
        Transformed tensor.
    """
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


def tanh_saturation(x, b: float = 0.5, c: float = 0.5):
    R"""Tanh saturation transformation.

    .. math::
        f(x) = b \tanh \left( \frac{x}{bc} \right)

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import tanh_saturation
        plt.style.use('arviz-darkgrid')
        params = [
            (0.75, 0.25),
            (0.75, 1.5),
            (1, 0.25),
            (1, 1),
            (1, 1.5),
        ]
        x = np.linspace(0, 5, 100)
        ax = plt.subplot(111)
        for b, c in params:
            y = tanh_saturation(x, b=b, c=c).eval()
            plt.plot(x, y, label=f'b = {b}\nc = {c}')
        plt.xlabel('spend', fontsize=12)
        plt.ylabel('f(spend)', fontsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.legend()
        plt.show()

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
