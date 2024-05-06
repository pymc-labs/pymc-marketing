#   Copyright 2024 The PyMC Labs Developers
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
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pytensor.tensor.random.op import RandomVariable

__all__ = [
    "ContContract",
    "ContNonContract",
    "ParetoNBD",
]


class ContNonContractRV(RandomVariable):
    name = "continuous_non_contractual"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ContNonContract", "\\operatorname{ContNonContract}")

    def make_node(self, rng, size, dtype, lam, p, T):
        T = pt.as_tensor_variable(T)

        return super().make_node(rng, size, dtype, lam, p, T)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, size):
        size = pm.distributions.shape_utils.to_tuple(size)

        # TODO: broadcast sizes
        lam = np.asarray(lam)
        p = np.asarray(p)
        T = np.asarray(T)

        if size == ():
            size = np.broadcast_shapes(lam.shape, p.shape, T.shape)

        lam = np.broadcast_to(lam, size)
        p = np.broadcast_to(p, size)
        T = np.broadcast_to(T, size)

        x_1 = rng.poisson(lam * T)
        x_2 = rng.geometric(p)
        x = np.minimum(x_1, x_2)

        nzp = x == 0  # nzp = non-zero purchases

        if x.shape == ():
            if nzp:
                return np.array([0, 0])
            else:
                return np.array([rng.beta(x, np.maximum(x_1 + 1 - x_2, 1)) * T, x])

        x[nzp] = 1.0  # temporary to avoid errors in rng.beta below
        t_x = rng.beta(x, np.maximum(x_1 + 1 - x_2, 1)) * T

        x[nzp] = 0.0
        t_x[nzp] = 0.0

        return np.stack([t_x, x], axis=-1)

    def _supp_shape_from_params(*args, **kwargs):
        return (2,)


continuous_non_contractual = ContNonContractRV()


class ContNonContract(PositiveContinuous):
    r"""
    Individual-level model for the customer lifetime value. See equation (3)
    from Fader et al. (2005) [1]_.

    .. math::

        f(\lambda, p | x, t_1, \dots, t_x, T)
        = f(\lambda, p | t_x, T) = (1 - p)^x \lambda^x \exp(-\lambda T)
          + \delta_{x > 0} p (1 - p)^{x-1} \lambda^x \exp(-\lambda t_x)

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p] = \frac{1}{p} - \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`
    ========  ===============================================

    References
    ----------
    .. [1] Fader, Peter S., Bruce GS Hardie, and Ka Lok Lee. "“Counting your customers”
           the easy way: An alternative to the Pareto/NBD model." Marketing science
           24.2 (2005): 275-284.
    """

    rv_op = continuous_non_contractual

    @classmethod
    def dist(cls, lam, p, T, **kwargs):
        return super().dist([lam, p, T], **kwargs)

    def logp(value, lam, p, T):
        t_x = value[..., 0]
        x = value[..., 1]

        zero_observations = pt.eq(x, 0)

        A = x * pt.log(1 - p) + x * pt.log(lam) - lam * T
        B = pt.log(p) + (x - 1) * pt.log(1 - p) + x * pt.log(lam) - lam * t_x

        logp = pt.switch(
            zero_observations,
            A,
            pt.logaddexp(A, B),
        )

        logp = pt.switch(
            pt.any(
                (
                    pt.and_(pt.ge(t_x, 0), zero_observations),
                    pt.lt(t_x, 0),
                    pt.lt(x, 0),
                    pt.gt(t_x, T),
                ),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            lam > 0,
            0 <= p,
            p <= 1,
            msg="lam > 0, 0 <= p <= 1",
        )


class ContContractRV(RandomVariable):
    name = "continuous_contractual"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ContinuousContractual", "\\operatorname{ContinuousContractual}")

    def make_node(self, rng, size, dtype, lam, p, T):
        T = pt.as_tensor_variable(T)

        return super().make_node(rng, size, dtype, lam, p, T)

    def __call__(self, lam, p, T, size=None, **kwargs):
        return super().__call__(lam, p, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, size):
        size = pm.distributions.shape_utils.to_tuple(size)

        # To do: broadcast sizes
        lam = np.asarray(lam)
        p = np.asarray(p)
        T = np.asarray(T)

        if size == ():
            size = np.broadcast_shapes(lam.shape, p.shape, T.shape)

        lam = np.broadcast_to(lam, size)
        p = np.broadcast_to(p, size)
        T = np.broadcast_to(T, size)

        x_1 = rng.poisson(lam * T)
        x_2 = rng.geometric(p)
        x = np.minimum(x_1, x_2)

        nzp = x == 0  # nzp = non-zero purchases

        if x.shape == ():
            if nzp:
                return np.array([0, 0, float(x_1 > x_2)])
            else:
                return np.array(
                    [rng.beta(x, np.maximum(x_1 + 1 - x_2, 1)) * T, x, float(x_1 > x_2)]
                )

        x[nzp] = 1.0  # temporary to avoid errors in rng.beta below
        t_x = rng.beta(x, np.maximum(x_1 + 1 - x_2, 1)) * T

        x[nzp] = 0.0
        t_x[nzp] = 0.0

        return np.stack([t_x, x, (x_1 > x_2).astype(float)], axis=-1)

    def _supp_shape_from_params(*args, **kwargs):
        return (3,)


continuous_contractual = ContContractRV()


class ContContract(PositiveContinuous):
    r"""
    Distribution class of a continuous contractual data-generating process,
    that is where purchases can occur at any time point (continuous) and
    churning/dropping out is explicit (contractual).

    .. math::

        f(\lambda, p | d, x, t_1, \dots, t_x, T)
        = f(\lambda, p | t_x, T) = (1 - p)^{x-1} \lambda^x \exp(-\lambda t_x)
        p^d \left\{(1-p)\exp(-\lambda*(T - t_x))\right\}^{1 - d}

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p, d] = \frac{1}{p} - \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`
    ========  ===============================================

    """

    rv_op = continuous_contractual

    @classmethod
    def dist(cls, lam, p, T, **kwargs):
        return super().dist([lam, p, T], **kwargs)

    def logp(value, lam, p, T):
        t_x = value[..., 0]
        x = value[..., 1]
        churn = value[..., 2]

        zero_observations = pt.eq(x, 0)

        logp = (x - 1) * pt.log(1 - p) + x * pt.log(lam) - lam * t_x
        logp += churn * pt.log(p) + (1 - churn) * (pt.log(1 - p) - lam * (T - t_x))

        logp = pt.switch(
            zero_observations,
            -lam * T,
            logp,
        )

        logp = pt.switch(
            pt.any(pt.or_(pt.lt(t_x, 0), zero_observations)),
            -np.inf,
            logp,
        )
        logp = pt.switch(
            pt.all(
                pt.or_(pt.eq(churn, 0), pt.eq(churn, 1)),
            ),
            logp,
            -np.inf,
        )
        logp = pt.switch(
            pt.any(
                (
                    pt.lt(t_x, 0),
                    pt.lt(x, 0),
                    pt.gt(t_x, T),
                ),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            lam > 0,
            0 <= p,
            p <= 1,
            pt.all(0 < T),
            msg="lam > 0, 0 <= p <= 1",
        )


class ParetoNBDRV(RandomVariable):
    name = "pareto_nbd"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ParetoNBD", "\\operatorname{ParetoNBD}")

    def make_node(self, rng, size, dtype, r, alpha, s, beta, T):
        r = pt.as_tensor_variable(r)
        alpha = pt.as_tensor_variable(alpha)
        s = pt.as_tensor_variable(s)
        beta = pt.as_tensor_variable(beta)
        T = pt.as_tensor_variable(T)

        return super().make_node(rng, size, dtype, r, alpha, s, beta, T)

    def __call__(self, r, alpha, s, beta, T, size=None, **kwargs):
        return super().__call__(r, alpha, s, beta, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, r, alpha, s, beta, T, size):
        size = pm.distributions.shape_utils.to_tuple(size)

        r = np.asarray(r)
        alpha = np.asarray(alpha)
        s = np.asarray(s)
        beta = np.asarray(beta)
        T = np.asarray(T)

        if size == ():
            size = np.broadcast_shapes(
                r.shape, alpha.shape, s.shape, beta.shape, T.shape
            )

        r = np.broadcast_to(r, size)
        alpha = np.broadcast_to(alpha, size)
        s = np.broadcast_to(s, size)
        beta = np.broadcast_to(beta, size)
        T = np.broadcast_to(T, size)

        output = np.zeros(shape=size + (2,))  # noqa:RUF005

        lam = rng.gamma(shape=r, scale=1 / alpha, size=size)
        mu = rng.gamma(shape=s, scale=1 / beta, size=size)

        def sim_data(lam, mu, T):
            t = 0
            n = 0

            dropout_time = rng.exponential(scale=1 / mu)
            wait = rng.exponential(scale=1 / lam)

            final_t = min(dropout_time, T)
            while (t + wait) < final_t:
                t += wait
                n += 1
                wait = rng.exponential(scale=1 / lam)

            return np.array(
                [
                    t,
                    n,
                ],
            )

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], mu[index], T[index])

        return output

    def _supp_shape_from_params(*args, **kwargs):
        return (2,)


pareto_nbd = ParetoNBDRV()


class ParetoNBD(PositiveContinuous):
    r"""
    Population-level distribution class for a continuous, non-contractual, Pareto/NBD process,
    based on Schmittlein, et al. in [2]_.

    The likelihood function is derived from equations (22) and (23) of [3]_, with terms
    rearranged for numerical stability.

    The modified expression is provided below:

    .. math::

        \begin{align}
        \text{if }\alpha > \beta: \\
        \\
        \mathbb{L}(r, \alpha, s, \beta | x, t_x, T) &=
        \frac{\Gamma(r+x)\alpha^r\beta}{\Gamma(r)+(\alpha +t_x)^{r+s+x}}
        [(\frac{s}{r+s+x})_2F_1(r+s+x,s+1;r+s+x+1;\frac{\alpha-\beta}{\alpha+t_x}) \\
        &+ (\frac{r+x}{r+s+x})
        \frac{_2F_1(r+s+x,s;r+s+x+1;\frac{\alpha-\beta}{\alpha+T})(\alpha +t_x)^{r+s+x}}
        {(\alpha +T)^{r+s+x}}] \\
        \\
        \text{if }\beta >= \alpha: \\
        \\
        \mathbb{L}(r, \alpha, s, \beta | x, t_x, T) &=
        \frac{\Gamma(r+x)\alpha^r\beta}{\Gamma(r)+(\beta +t_x)^{r+s+x}}
        [(\frac{s}{r+s+x})_2F_1(r+s+x,r+x;r+s+x+1;\frac{\beta-\alpha}{\beta+t_x}) \\
        &+ (\frac{r+x}{r+s+x})
        \frac{_2F_1(r+s+x,r+x+1;r+s+x+1;\frac{\beta-\alpha}{\beta+T})(\beta +t_x)^{r+s+x}}
        {(\beta +T)^{r+s+x}}]
        \end{align}

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | r, \alpha, s, \beta] = \frac{r\beta}{\alpha(s-1)}[1-(\frac{\beta}{\beta + t})^{s-1}]`
    ========  ===============================================

    References
    ----------
    .. [2] David C. Schmittlein, Donald G. Morrison and Richard Colombo.
           "Counting Your Customers: Who Are They and What Will They Do Next."
           Management Science,Vol. 33, No. 1 (Jan., 1987), pp. 1-24.

    .. [3] Fader, Peter & G. S. Hardie, Bruce (2005).
           "A Note on Deriving the Pareto/NBD Model and Related Expressions."
           http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf
    """  # noqa: E501

    rv_op = pareto_nbd

    @classmethod
    def dist(cls, r, alpha, s, beta, T, **kwargs):
        return super().dist([r, alpha, s, beta, T], **kwargs)

    def logp(value, r, alpha, s, beta, T):
        t_x = value[..., 0]
        x = value[..., 1]

        rsx = r + s + x
        rx = r + x

        cond = alpha >= beta
        larger_param = pt.switch(cond, alpha, beta)
        smaller_param = pt.switch(cond, beta, alpha)
        param_diff = larger_param - smaller_param
        hyp2f1_t1_2nd_param = pt.switch(cond, s + 1, rx)
        hyp2f1_t2_2nd_param = pt.switch(cond, s, rx + 1)

        # This term is factored out of the denominator of hyp2f_t1 for numerical stability
        refactored = rsx * pt.log(larger_param + t_x)

        hyp2f1_t1 = pt.log(
            pt.hyp2f1(
                rsx, hyp2f1_t1_2nd_param, rsx + 1, param_diff / (larger_param + t_x)
            )
        )
        hyp2f1_t2 = (
            pt.log(
                pt.hyp2f1(
                    rsx, hyp2f1_t2_2nd_param, rsx + 1, param_diff / (larger_param + T)
                )
            )
            - rsx * pt.log(larger_param + T)
            + refactored
        )

        A1 = (
            pt.gammaln(rx)
            - pt.gammaln(r)
            + r * pt.log(alpha)
            + s * pt.log(beta)
            - refactored
        )
        A2 = pt.log(s) - pt.log(rsx) + hyp2f1_t1
        A3 = pt.log(rx) - pt.log(rsx) + hyp2f1_t2

        logp = A1 + pt.logaddexp(A2, A3)

        logp = pt.switch(
            pt.or_(
                pt.or_(
                    pt.lt(t_x, 0),
                    pt.lt(x, 0),
                ),
                pt.gt(t_x, T),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            r > 0,
            alpha > 0,
            s > 0,
            beta > 0,
            msg="r > 0, alpha > 0, s > 0, beta > 0",
        )
