#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Media transformation functions for Marketing Mix Models."""

from enum import Enum
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pytensor.npy_2_compat import normalize_axis_index


class ConvMode(str, Enum):
    """Convolution mode for the convolution."""

    # TODO: use StrEnum when we upgrade to python 3.11
    After = "After"
    Before = "Before"
    Overlap = "Overlap"


class WeibullType(str, Enum):
    """Weibull type for the Weibull adstock."""

    # TODO: use StrEnum when we upgrade to python 3.11
    PDF = "PDF"
    CDF = "CDF"


def batched_convolution(
    x,
    w,
    axis: int = 0,
    mode: ConvMode | str = ConvMode.After,
):
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
            suffix = "\n(default)" if mode == ConvMode.After else ""
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
    x : tensor_like
        The array to convolve.
    w : tensor_like
        The weight of the convolution. The last axis of ``w`` determines the number of steps
        to use in the convolution.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.After.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
          similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
          where the effect overlaps with both preceding and succeeding elements.

    Returns
    -------
    y : tensor_like
        The result of convolving ``x`` with ``w`` along the desired axis. The shape of the
        result will match the shape of ``x`` up to broadcasting with ``w``. The convolved
        axis will show the results of left padding zeros to ``x`` while applying the
        convolutions.

    """
    # We move the axis to the last dimension of the array so that it's easier to
    # reason about parameter broadcasting. We will move the axis back at the end
    x = pt.as_tensor(x)
    w = pt.as_tensor(w)

    axis = normalize_axis_index(axis, x.ndim)
    x = pt.moveaxis(x, axis, -1)
    x_batch_shape = tuple(x.shape)[:-1]
    lags = w.shape[-1]

    if mode == ConvMode.After:
        zeros = pt.zeros((*x_batch_shape, lags - 1), dtype=x.dtype)
        padded_x = pt.join(-1, zeros, x)
    elif mode == ConvMode.Before:
        zeros = pt.zeros((*x_batch_shape, lags - 1), dtype=x.dtype)
        padded_x = pt.join(-1, x, zeros)
    elif mode == ConvMode.Overlap:
        zeros_left = pt.zeros((*x_batch_shape, lags // 2), dtype=x.dtype)
        zeros_right = pt.zeros((*x_batch_shape, (lags - 1) // 2), dtype=x.dtype)
        padded_x = pt.join(-1, zeros_left, x, zeros_right)
    else:
        raise ValueError(f"Wrong Mode: {mode}, expected one of {', '.join(ConvMode)}")

    conv = pt.signal.convolve1d(padded_x, w, mode="valid")
    return pt.moveaxis(conv, -1, axis + conv.ndim - x.ndim)


def geometric_adstock(
    x,
    alpha: float = 0.0,
    l_max: int = 12,
    normalize: bool = False,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
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
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.After.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling
       with carryover and shape effects." (2017).

    """
    alpha = check_parameters(
        alpha, [pt.ge(alpha, 0), pt.le(alpha, 1)], msg="0 <= alpha <= 1"
    )

    w = pt.power(pt.as_tensor(alpha)[..., None], pt.arange(l_max, dtype=x.dtype))
    w = w / pt.sum(w, axis=-1, keepdims=True) if normalize else w
    return batched_convolution(x, w, axis=axis, mode=mode)


def delayed_adstock(
    x,
    alpha: float = 0.0,
    theta: int = 0,
    l_max: int = 12,
    normalize: bool = False,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
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
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.After.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.

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
    return batched_convolution(x, w, axis=axis, mode=mode)


def weibull_adstock(
    x,
    lam=1,
    k=1,
    l_max: int = 12,
    axis: int = 0,
    mode: ConvMode = ConvMode.After,
    type: WeibullType | str = WeibullType.PDF,
    normalize: bool = False,
):
    R"""Weibull Adstocking Transformation.

    This transformation is similar to geometric adstock transformation but has more
    degrees of freedom, adding more flexibility.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import WeibullType, weibull_adstock
        plt.style.use('arviz-darkgrid')

        spend = np.zeros(50)
        spend[0] = 1

        shapes = [0.5, 1., 1.5, 5.]
        scales = [10, 20, 40]
        modes = [WeibullType.PDF, WeibullType.CDF]

        fig, axes = plt.subplots(
            len(shapes), len(modes), figsize=(12, 8), sharex=True, sharey=True
        )
        fig.suptitle("Effect of Changing Weibull Adstock Parameters", fontsize=16)

        for m, mode in enumerate(modes):
            axes[0, m].set_title(f"Mode: {mode.value}")

            for i, shape in enumerate(shapes):
                for j, scale in enumerate(scales):
                    adstock = weibull_adstock(
                        spend, lam=scale, k=shape, type=mode, l_max=len(spend)
                    ).eval()

                    axes[i, m].plot(
                        np.arange(len(spend)),
                        adstock,
                        label=f"Scale={scale}",
                        linestyle="-",
                    )

        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="center right",
            bbox_to_anchor=(1.2, 0.85),
        )

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()



    Parameters
    ----------
    x : tensor
        Input tensor.
    lam : float, by default 1.
        Scale parameter of the Weibull distribution. Must be positive.
    k : float, by default 1.
        Shape parameter of the Weibull distribution. Must be positive.
    l_max : int, by default 12
        Maximum duration of carryover effect.
    axis : int
        The axis of ``x`` along witch to apply the convolution
    mode : ConvMode, optional
        The convolution mode determines how the convolution is applied at the boundaries
        of the input signal, denoted as "x." The default mode is ConvMode.After.

        - ConvMode.After: Applies the convolution with the "Adstock" effect, resulting in a trailing decay effect.
        - ConvMode.Before: Applies the convolution with the "Excitement" effect, creating a leading effect
            similar to the wow factor.
        - ConvMode.Overlap: Applies the convolution with both "Pull-Forward" and "Pull-Backward" effects,
            where the effect overlaps with both preceding and succeeding elements.
    type : WeibullType or str, by default WeibullType.PDF
        Type of Weibull adstock transformation to be applied (PDF or CDF).
    normalize : bool, by default False
        Whether to normalize the weights.


    Returns
    -------
    tensor
        Transformed tensor based on Weibull adstock transformation.

    """
    lam = pt.as_tensor(lam)[..., None]
    k = pt.as_tensor(k)[..., None]
    t = pt.arange(l_max, dtype=x.dtype) + 1

    if type == WeibullType.PDF:
        w = pt.exp(pm.Weibull.logp(t, k, lam))
        w = (w - pt.min(w, axis=-1)[..., None]) / (
            pt.max(w, axis=-1)[..., None] - pt.min(w, axis=-1)[..., None]
        )
    elif type == WeibullType.CDF:
        w = 1 - pt.exp(pm.Weibull.logcdf(t, k, lam))
        shape = (*w.shape[:-1], w.shape[-1] + 1)
        padded_w = pt.ones(shape, dtype=w.dtype)
        padded_w = pt.set_subtensor(padded_w[..., 1:], w)
        w = pt.cumprod(padded_w, axis=-1)
    else:
        raise ValueError(f"Wrong WeibullType: {type}, expected of WeibullType")

    w = w / pt.sum(w, axis=-1, keepdims=True) if normalize else w
    return batched_convolution(x, w, axis=axis, mode=mode)


def logistic_saturation(x, lam: npt.NDArray | float = 0.5):
    r"""Logistic saturation transformation.

    .. math::
        f(x) = \frac{1 - e^{-\lambda x}}{1 + e^{-\lambda x}}

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


def inverse_scaled_logistic_saturation(
    x, lam: npt.NDArray | float = 0.5, eps: float = np.log(3)
):
    r"""Inverse scaled logistic saturation transformation.

    It offers a more intuitive alternative to logistic_saturation,
    allowing for lambda to be interpreted as the half saturation point
    when using default value for eps.

    .. math::
        f(x) = \frac{1 - e^{-x*\epsilon/\lambda}}{1 + e^{-x*\epsilon/\lambda}}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import inverse_scaled_logistic_saturation
        plt.style.use('arviz-darkgrid')
        lam = np.array([0.25, 0.5, 1, 2, 4])
        x = np.linspace(0, 5, 100)
        ax = plt.subplot(111)
        for l in lam:
            y = inverse_scaled_logistic_saturation(x, lam=l).eval()
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
    eps : float or array-like, optional, by default ln(3)
        Scaling parameter. ln(3) results in halfway saturation at lam

    Returns
    -------
    tensor
        Transformed tensor.

    """
    return logistic_saturation(x, eps / lam)


class TanhSaturationParameters(NamedTuple):
    """Container for tanh saturation parameters.

    Parameters
    ----------
    b : pt.TensorLike
        Saturation
    c : pt.TensorLike
        Customer Aquisition Cost at 0.

    """

    b: pt.TensorLike
    c: pt.TensorLike

    def baseline(self, x0: pt.TensorLike) -> "TanhSaturationBaselinedParameters":
        """Change the parameterization to baselined at :math:`x_0`.

        Parameters
        ----------
        x0 : pt.TensorLike
            Baseline spend.

        Returns
        -------
        TanhSaturationBaselinedParameters
            Baselined parameters.

        """
        y_ref = tanh_saturation(x0, self.b, self.c)
        gain_ref = y_ref / x0
        r_ref = y_ref / self.b
        return TanhSaturationBaselinedParameters(x0, gain_ref, r_ref)


class TanhSaturationBaselinedParameters(NamedTuple):
    """Representation of tanh saturation parameters in baselined form.

    Parameters
    ----------
    x0 : pt.TensorLike
        Baseline spend.
    gain : pt.TensorLike
        ROAS at :math:`x_0`.
    r : pt.TensorLike
        Overspend Fraction.

    """

    x0: pt.TensorLike
    gain: pt.TensorLike
    r: pt.TensorLike

    def debaseline(self) -> TanhSaturationParameters:
        """Change the parameterization to baselined to be classic saturation and cac.

        Returns
        -------
        TanhSaturationParameters
            Classic saturation and cac parameters.

        """
        saturation = (self.gain * self.x0) / self.r
        cac = self.r / (self.gain * pt.arctanh(self.r))
        return TanhSaturationParameters(saturation, cac)

    def rebaseline(self, x1: pt.TensorLike) -> "TanhSaturationBaselinedParameters":
        """Change the parameterization to baselined at :math:`x_1`."""
        params = self.debaseline()
        return params.baseline(x1)


def tanh_saturation(
    x: pt.TensorLike,
    b: pt.TensorLike = 0.5,
    c: pt.TensorLike = 0.5,
) -> pt.TensorVariable:
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
    See https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/ # noqa: E501

    """  # noqa: E501
    return b * pt.tanh(x / (b * c))


def tanh_saturation_baselined(
    x: pt.TensorLike,
    x0: pt.TensorLike,
    gain: pt.TensorLike = 0.5,
    r: pt.TensorLike = 0.5,
) -> pt.TensorVariable:
    r"""Baselined Tanh Saturation.

    This parameterization that is easier than :func:`tanh_saturation`
    to use for industry applications where domain knowledge is an essence.

    In a nutshell, it is an alternative parameterization of the reach function is given by:

    .. math::

        \begin{align}
        c_0 &= \frac{r}{g \cdot \arctan(r)} \\
        \beta &= \frac{g \cdot x_0}{r} \\
        \operatorname{saturation}(x, \beta, c_0) &= \beta  \cdot \tanh \left( \frac{x}{c_0 \cdot \beta} \right)
        \end{align}

    where:

    - :math:`x_0` is the "reference point". This is a point chosen
      by the user (not given a prior) where they expect most of their data to lie.
      For example, if you're spending between 50 and 150 dollars on a particular channel,
      you might choose :math:`x_0 = 100`.
      Suggested value is median channel spend: ``np.median(spend)``.

    - :math:`g` is the "gain", which is the value of the CAC (:math:`c_0`) at the reference point.
      You have to set a prior on what you think the CAC is when you spend :math:`x_0 = 100`.
      Imagine you have four advertising channels, and you acquired 1000 new users.
      If each channel performed equally well, and advertising drove all sales, you might expect
      that you gained 250 users from each channel.  Here, your "gain" would be :math:`250 / 100 = 2.5`.
      Suggested prior is ``pm.Exponential``
    - :math:`r`, the overspend fraction is telling you where the reference point is.

      - :math:`0` - we can increase our budget by a lot to reach the saturated region,
        the diminishing returns are not visible yet.
      - :math:`1` - the reference point is already in the saturation region
        and additional dollar spend will not lead to any new users.
      - :math:`0.8`, you can still increase acquired users by :math:`50\%` as much
        you get in the reference point by increasing the budget.
        :math:`x_0` effect is 20% away from saturation point

      Suggested prior is ``pm.Beta``

    .. note::

        The reference point :math:`x_0` has to be set within the range of the actual spends.
        As in, you buy ads three times and spend :math:`5`, :math:`6` and :math:`7` dollars,
        :math:`x_0` has to be set within :math:`[5, 7]`, so not :math:`4` not :math:`8`.
        Otherwise the posterior of r and gain becomes a skinny diagonal line.
        It could be very relevant if there is very little spend observations for a particular channel.

    The original reach or saturation function used in an MMM is formulated as

    .. math::

        \operatorname{saturation}(x, \beta, c_0) = \beta  \cdot \tanh \left( \frac{x}{c_0 \cdot \beta} \right)

    where:

    - :math:`\beta` is the saturation, or the limit of the total number
      of new users obtained when an infinite number of dollars are spent on that channel.
    - :math:`c_0` is the cost per acquisition (CAC0), so the initial cost per new user.
    - :math:`\frac{1}{c_0}` is the inverse of the CAC0, so it's the number of new
      users we might expect after spending our first dollar.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import (
            tanh_saturation_baselined,
            tanh_saturation,
            TanhSaturationBaselinedParameters,
        )

        gain = 1
        overspend_fraction = 0.7
        x_baseline = 400

        params = TanhSaturationBaselinedParameters(x_baseline, gain, overspend_fraction)

        x = np.linspace(0, 1000)
        y = tanh_saturation_baselined(x, *params).eval()

        saturation, cac0 = params.debaseline()
        cac0 = cac0.eval()
        saturated_ref = tanh_saturation(x_baseline, saturation, cac0).eval()

        plt.plot(x, y);
        plt.axvline(x_baseline, linestyle="dashed", color="red", label="baseline")
        plt.plot(x, x * gain, linestyle="dashed", label="gain (slope)");
        plt.axhline(saturated_ref, linestyle="dashed", label="f(reference)")
        plt.plot(x, x / cac0, linestyle="dotted", label="1/cac (slope)");
        plt.axhline(saturation, linestyle="dotted", label="saturation")
        plt.fill_between(x, saturated_ref, saturation, alpha=0.1, label="underspend fraction")
        plt.fill_between(x, saturated_ref, alpha=0.1, label="overspend fraction")
        plt.legend()
        plt.show()

    Examples
    --------

    .. code-block:: python

        import pymc as pm
        import numpy as np

        x_in = np.exp(3+np.random.randn(100))
        true_cac = 1
        true_saturation = 100
        y_out = abs(np.random.normal(tanh_saturation(x_in, true_saturation, true_cac).eval(), 0.1))

        with pm.Model() as model_reparam:
            r = pm.Uniform("r")
            gain = pm.Exponential("gain", 1)
            input = pm.ConstantData("spent", x_in)
            response = pm.ConstantData("response", y_out)
            sigma = pm.HalfNormal("n")
            output = tanh_saturation_baselined(input, np.median(x_in), gain, r)
            pm.Normal("output", output, sigma, observed=response)
            trace = pm.sample()

    Parameters
    ----------
    x : tensor
        Input tensor.
    x0: tensor
        Baseline for saturation.
    gain : tensor, by default 0.5
        ROAS at the baseline point, mathematically as :math:`gain = f(x0) / x0`.
    r : tensor, by default 0.5
        The overspend fraction, mathematically as :math:`r = f(x0) / \text{saturation}`.

    Returns
    -------
    tensor
        Transformed tensor.

    References
    ----------
    Developed by Max Kochurov and Aziz Al-Maeeni doing innovative work in `PyMC Labs <pymc-labs.com>`_.

    """
    return gain * x0 * pt.tanh(x * pt.arctanh(r) / x0) / r


def michaelis_menten(
    x: float | np.ndarray | npt.NDArray,
    alpha: float | np.ndarray | npt.NDArray,
    lam: float | np.ndarray | npt.NDArray,
) -> float | Any:
    r"""Evaluate the Michaelis-Menten function for given values of x, alpha, and lambda.

    The Michaelis-Menten function models enzyme kinetics and describes how the rate of
    a chemical reaction increases with substrate concentration until it reaches its
    maximum value.

    .. math::
        \alpha \cdot \frac{x}{\lambda + x}

    where:
     - :math:`x`: Channel spend or substrate concentration.
     - :math:`\alpha`: Maximum contribution or efficiency factor.
     - :math:`\lambda` (k): Michaelis constant, representing the threshold substrate concentration.

    .. plot::
        :context: close-figs

        import numpy as np
        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.transformers import michaelis_menten

        x = np.linspace(0, 100, 500)
        alpha = 10
        lam = 50
        y = michaelis_menten(x, alpha, lam)

        plt.plot(x, y)
        plt.xlabel('Spend/Impressions (x)')
        plt.ylabel('Contribution (y)')
        plt.title('Michaelis-Menten Function')
        plt.show()

    .. plot::
        :context: close-figs

        import numpy as np
        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.transformers import michaelis_menten

        x = np.linspace(0, 100, 500)
        alpha_values = [5, 10, 15]  # Different values of alpha
        lam_values = [25, 50, 75]  # Different values of lam

        # Plot varying lam
        plt.figure(figsize=(8, 6))
        for lam in lam_values:
            y = michaelis_menten(x, alpha_values[0], lam)
            plt.plot(x, y, label=f"lam={lam}")
        plt.xlabel('Spend/Impressions (x)')
        plt.ylabel('Contribution (y)')
        plt.title('Michaelis-Menten Function (Varying lam)')
        plt.legend()
        plt.show()

        # Plot varying alpha
        plt.figure(figsize=(8, 6))
        for alpha in alpha_values:
            y = michaelis_menten(x, alpha, lam_values[0])
            plt.plot(x, y, label=f"alpha={alpha}")
        plt.xlabel('Spend/Impressions (x)')
        plt.ylabel('Contribution (y)')
        plt.title('Michaelis-Menten Function (Varying alpha)')
        plt.legend()
        plt.show()

    Parameters
    ----------
    x : float
        The spent on a channel.
    alpha : float
        The maximum contribution a channel can make.
    lam : float
        The Michaelis constant for the given enzyme-substrate system.

    Returns
    -------
    float
        The value of the Michaelis-Menten function given the parameters.

    """
    return alpha * x / (lam + x)


def hill_function(
    x: pt.TensorLike, slope: pt.TensorLike, kappa: pt.TensorLike
) -> pt.TensorVariable:
    r"""Hill Function.

    .. math::
        f(x) = 1 - \frac{\kappa^s}{\kappa^s + x^s}

    where:
     - :math:`s` is the slope of the hill.
     - :math:`\kappa` is the half-saturation point as :math:`f(\kappa) = 0.5` for any value of :math:`s` and :math:`\kappa`.
     - :math:`x` is the independent variable and must be non-negative.

    Hill function from Equation (5) in the paper [1]_.

    .. plot::
        :context: close-figs

        import numpy as np
        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.transformers import hill_function
        x = np.linspace(0, 10, 100)
        # Varying slope
        slopes = [0.3, 0.7, 1.2]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, slope in enumerate(slopes):
            plt.subplot(1, 3, i+1)
            y = hill_function(x, slope, 2).eval()
            plt.plot(x, y)
            plt.xlabel('x')
            plt.title(f'Slope = {slope}')
        plt.subplot(1,3,1)
        plt.ylabel('Hill Saturation Sigmoid')
        plt.tight_layout()
        plt.show()
        # Varying kappa
        kappas = [1, 5, 10]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, kappa in enumerate(kappas):
            plt.subplot(1, 3, i+1)
            y = hill_function(x, 1, kappa).eval()
            plt.plot(x, y)
            plt.xlabel('x')
            plt.title(f'Kappa = {kappa}')
        plt.subplot(1,3,1)
        plt.ylabel('Hill Saturation Sigmoid')
        plt.tight_layout()
        plt.show()

    Parameters
    ----------
    x : float or array-like
        The independent variable, typically representing the concentration of a
        substrate or the intensity of a stimulus.
    slope : float
        The slope of the hill. Must pe non-positive.
    kappa : float
        The half-saturation point as :math:`f(\kappa) = 0.5` for any value of :math:`s` and :math:`\kappa`.

    Returns
    -------
    float
        The value of the Hill function given the parameters.

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).

    """  # noqa: E501
    return pt.as_tensor_variable(
        1 - pt.power(kappa, slope) / (pt.power(kappa, slope) + pt.power(x, slope))
    )


def hill_saturation_sigmoid(
    x: pt.TensorLike,
    sigma: pt.TensorLike,
    beta: pt.TensorLike,
    lam: pt.TensorLike,
) -> pt.TensorVariable:
    r"""Hill Saturation Sigmoid Function.

    .. math::
        f(x) = \frac{\sigma}{1 + e^{-\beta(x - \lambda)}} - \frac{\sigma}{1 + e^{\beta\lambda}}

    where:
     - :math:`\sigma` is the upper asymptote
     - :math:`\beta` is the slope parameter
     - :math:`\lambda` is the transition point on the X-axis
     - :math:`x` is the independent variable

    This function computes the Hill sigmoidal response curve, which is commonly
    used to describe the saturation effect in biological systems. The curve is
    characterized by its sigmoidal shape, representing a gradual transition from
    a low, nearly zero level to a high plateau, the maximum value the function
    will approach as the independent variable grows large. In this implementation,
    we add an offset to the sigmoid function to ensure that the function always passes
    through the origin as we expect zero spend to result in zero contribution.

    .. plot::
        :context: close-figs

        import numpy as np
        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.transformers import hill_saturation_sigmoid
        x = np.linspace(0, 10, 100)
        # Varying sigma
        sigmas = [0.5, 1, 1.5]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, sigma in enumerate(sigmas):
            plt.subplot(1, 3, i+1)
            y = hill_saturation_sigmoid(x, sigma, 2, 5).eval()
            plt.plot(x, y)
            plt.xlabel('x')
            plt.title(f'Sigma = {sigma}')
        plt.subplot(1,3,1)
        plt.ylabel('Hill Saturation Sigmoid')
        plt.tight_layout()
        plt.show()
        # Varying beta
        betas = [1, 2, 3]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, beta in enumerate(betas):
            plt.subplot(1, 3, i+1)
            y = hill_saturation_sigmoid(x, 1, beta, 5).eval()
            plt.plot(x, y)
            plt.xlabel('x')
            plt.title(f'Beta = {beta}')
        plt.subplot(1,3,1)
        plt.ylabel('Hill Saturation Sigmoid')
        plt.tight_layout()
        plt.show()
        # Varying lam
        lams = [3, 5, 7]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for i, lam in enumerate(lams):
            plt.subplot(1, 3, i+1)
            y = hill_saturation_sigmoid(x, 1, 2, lam).eval()
            plt.plot(x, y)
            plt.xlabel('x')
            plt.title(f'Lambda = {lam}')
        plt.subplot(1,3,1)
        plt.ylabel('Hill Saturation Sigmoid')
        plt.tight_layout()
        plt.show()

    Parameters
    ----------
    x : float or array-like
        The independent variable, typically representing the concentration of a
        substrate or the intensity of a stimulus.
    sigma : float
        The upper asymptote of the curve, representing the approximate maximum value the
        function will approach as x grows large. The true maximum value is at `sigma * (1 - 1 / (1 + exp(beta * lam)))`
    beta : float
        The slope parameter, determining the steepness of the curve.
    lam : float
        The x-value of the midpoint where the curve transitions from exponential
        growth to saturation.

    Returns
    -------
    float or array-like
        The value of the Hill saturation sigmoid function for each input value of x.

    """
    return sigma / (1 + pt.exp(-beta * (x - lam))) - sigma / (1 + pt.exp(beta * lam))


def root_saturation(
    x: pt.TensorLike,
    alpha: pt.TensorLike,
) -> pt.TensorVariable:
    r"""Root saturation transformation.

    .. math::
        f(x) = x^{\alpha}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        from pymc_marketing.mmm.transformers import root_saturation
        plt.style.use('arviz-darkgrid')
        alpha = np.array([0.1, 0.3, 0.5, 0.7])
        x = np.linspace(0, 5, 100)
        ax = plt.subplot(111)
        for a in alpha:
            y = root_saturation(x, alpha=a)
            plt.plot(x, y, label=f'alpha = {a}')
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
    alpha : float
        Exponent for the root transformation. Must be non-negative.

    Returns
    -------
    tensor
        Transformed tensor.

    """
    return x**alpha
