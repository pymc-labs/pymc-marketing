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

    def make_node(self, rng, size, dtype, lam, p, T, T0):

        T = pt.as_tensor_variable(T)
        T0 = pt.as_tensor_variable(T0)

        return super().make_node(rng, size, dtype, lam, p, T, T0)

    def __call__(self, lam, p, T, T0=0, size=None, **kwargs):
        return super().__call__(lam, p, T, T0, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, T0, size) -> np.array:

        size = pm.distributions.shape_utils.to_tuple(size)

        # TODO: broadcast sizes
        lam = np.asarray(lam)
        p = np.asarray(p)
        T = np.asarray(T)
        T0 = np.asarray(T0)

        if size == ():
            size = np.broadcast_shapes(lam.shape, p.shape, T.shape, T0.shape)

        lam = np.broadcast_to(lam, size)
        p = np.broadcast_to(p, size)
        T = np.broadcast_to(T, size)
        T0 = np.broadcast_to(T0, size)

        output = np.zeros(shape=size + (2,))

        # TODO: Optimize to work in a vectorized manner!
        def sim_data(lam, p, T, T0):
            t = T0
            n = 0

            while True:
                wait = rng.exponential(scale=1 / lam)
                dropout = rng.binomial(n=1, p=p)

                if t + wait > T:
                    break
                else:
                    t += wait
                    n += 1

                    if dropout == 1:
                        break

            return np.array([t, n])

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index], T0[index])

        return output

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
    def dist(cls, lam, p, T, T0=0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]

        zero_observations = pt.eq(x, 0)

        A = x * pt.log(1 - p) + x * pt.log(lam) - lam * (T - T0)
        B = pt.log(p) + (x - 1) * pt.log(1 - p) + x * pt.log(lam) - lam * (t_x - T0)

        logp = pt.switch(
            zero_observations,
            A,
            pt.logaddexp(A, B),
        )

        logp = pt.switch(
            pt.any(
                (
                    pt.lt(t_x, T0),
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
            pt.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


class ContContractRV(RandomVariable):
    name = "continuous_contractual"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ContinuousContractual", "\\operatorname{ContinuousContractual}")

    def make_node(self, rng, size, dtype, lam, p, T, T0):

        T = pt.as_tensor_variable(T)
        T0 = pt.as_tensor_variable(T0)

        return super().make_node(rng, size, dtype, lam, p, T, T0)

    def __call__(self, lam, p, T, T0=0, size=None, **kwargs):
        return super().__call__(lam, p, T, T0, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, T0, size) -> np.array:

        size = pm.distributions.shape_utils.to_tuple(size)

        # To do: broadcast sizes
        lam = np.asarray(lam)
        p = np.asarray(p)
        T = np.asarray(T)
        T0 = np.asarray(T0)

        if size == ():
            size = np.broadcast_shapes(lam.shape, p.shape, T.shape, T0.shape)

        lam = np.broadcast_to(lam, size)
        p = np.broadcast_to(p, size)
        T = np.broadcast_to(T, size)
        T0 = np.broadcast_to(T0, size)

        output = np.zeros(shape=size + (3,))

        def sim_data(lam, p, T, T0):
            t = 0
            n = 0

            dropout = 0
            while not dropout:
                wait = rng.exponential(scale=1 / lam)
                # If we didn't go into the future
                if (t + wait) < T:
                    n += 1
                    t = t + wait
                    dropout = rng.binomial(n=1, p=p)
                else:
                    break

            return np.array(
                [
                    t,
                    n,
                    dropout,
                ],
            )

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index], T0[index])

        return output

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
    def dist(cls, lam, p, T, T0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]
        churn = value[..., 2]

        zero_observations = pt.eq(x, 0)

        logp = (x - 1) * pt.log(1 - p) + x * pt.log(lam) - lam * t_x
        logp += churn * pt.log(p) + (1 - churn) * (
            pt.log(1 - p) - lam * ((T - T0) - t_x)
        )

        logp = pt.switch(
            zero_observations,
            -lam * (T - T0),
            logp,
        )

        logp = pt.switch(
            pt.any(pt.or_(pt.lt(t_x, 0), pt.lt(x, 0))),
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
                    pt.lt(t_x, T0),
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
            pt.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


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
    def dist(cls, lam, p, T, T0=0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]

        zero_observations = pt.eq(x, 0)

        A = x * pt.log(1 - p) + x * pt.log(lam) - lam * (T - T0)
        B = pt.log(p) + (x - 1) * pt.log(1 - p) + x * pt.log(lam) - lam * (t_x - T0)

        logp = pt.switch(
            zero_observations,
            A,
            pt.logaddexp(A, B),
        )

        logp = pt.switch(
            pt.any(
                (
                    pt.lt(t_x, T0),
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
            pt.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


class ParetoNBDRV(RandomVariable):
    name = "pareto_nbd"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ParetoNBD", "\\operatorname{ParetoNBD}")

    def make_node(self, rng, size, dtype, r, alpha, s, beta, T, T0):

        T = pt.as_tensor_variable(T)
        T0 = pt.as_tensor_variable(T0)

        return super().make_node(rng, size, dtype, r, alpha, s, beta, T, T0)

    def __call__(self, r, alpha, s, beta, T, T0=0, size=None, **kwargs):
        return super().__call__(r, alpha, s, beta, T, T0, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, r, alpha, s, beta, T, T0, size) -> np.array:

        size = pm.distributions.shape_utils.to_tuple(size)

        r = np.asarray(r)
        alpha = np.asarray(alpha)
        s = np.asarray(s)
        beta = np.asarray(beta)
        T = np.asarray(T)
        T0 = np.asarray(T0)

        if size == ():
            size = np.broadcast_shapes(
                r.shape, alpha.shape, s.shape, beta.shape, T.shape, T0.shape
            )

        r = np.broadcast_to(r, size)
        alpha = np.broadcast_to(alpha, size)
        s = np.broadcast_to(s, size)
        beta = np.broadcast_to(beta, size)
        T = np.broadcast_to(T, size)
        T0 = np.broadcast_to(T0, size)

        output = np.zeros(shape=size + (2,))

        lam = rng.gamma(shape=r, scale=1 / alpha, size=size)
        mu = rng.gamma(shape=s, scale=1 / beta, size=size)

        def sim_data(lam, mu, T, T0):
            t = 0
            n = 0

            dropout_time = rng.exponential(scale=1 / mu)
            wait = rng.exponential(scale=1 / lam)

            while t + wait < min(dropout_time, T):
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
            output[index] = sim_data(lam[index], mu[index], T[index], T0[index])

        return output

    def _supp_shape_from_params(*args, **kwargs):
        return (3,)


pareto_nbd = ParetoNBDRV()


class ParetoNBD(PositiveContinuous):
    r"""
    Distribution class for a continuous, non-contractual, Pareto/NBD data-generating process.
    The Pareto/NBD model assumes transaction rates and lifetimes are both exponentially distributed,
    with heterogeneity following a gamma distribution.

    .. math::

        f(\lambda, p | d, x, t_1, \dots, t_x, T)
        = f(\lambda, p | t_x, T) = (1 - p)^{x-1} \lambda^x \exp(-\lambda t_x)
        p^d \left\{(1-p)\exp(-\lambda*(T - t_x))\right\}^{1 - d}

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p, d] = \frac{1}{p} - \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`
    ========  ===============================================

    """
    rv_op = pareto_nbd

    @classmethod
    def dist(cls, r, alpha, s, beta, T, T0, **kwargs):
        return super().dist([r, alpha, s, beta, T, T0], **kwargs)

    def logp(value, r, alpha, s, beta, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]

        age = T - T0
        rec = t_x - T0

        # Term A0 swaps alpha and beta terms depending on which is larger
        min_of_alpha_beta, max_of_alpha_beta, t = pt.switch(
            pt.lt(alpha, beta),
            (alpha, beta, r + x),
            (beta, alpha, s + 1),
        )

        abs_alpha_beta = max_of_alpha_beta - min_of_alpha_beta

        r_s_x = r + s + x

        p_1 = pt.hyp2f1(
            r_s_x, t, r_s_x + 1.0, abs_alpha_beta / (max_of_alpha_beta + rec)
        )
        q_1 = max_of_alpha_beta + rec
        p_2 = pt.hyp2f1(
            r_s_x, t, r_s_x + 1.0, abs_alpha_beta / (max_of_alpha_beta + age)
        )
        q_2 = max_of_alpha_beta + age

        # TODO: This will not converge properly because it must be subtracted rather than added!
        log_A_0 = pt.logaddexp(
            pt.log(p_1) + r_s_x * pt.log(q_2), pt.log(p_2) + r_s_x * pt.log(q_1)
        ) - r_s_x * pt.log(q_1 * q_2)

        A_1 = pt.gammaln(r + x) - pt.gammaln(r) + r * pt.log(alpha) + s * pt.log(beta)

        A_2 = pt.logaddexp(
            -(r + x) * pt.log(alpha + age) - s * pt.log(beta + age),
            pt.log(s) + log_A_0 - pt.log(r_s_x),
        )

        logp = A_1 + A_2

        logp = pt.switch(
            pt.any(
                (
                    pt.lt(t_x, T0),
                    pt.lt(x, 0),
                    pt.gt(t_x, T),
                ),
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
            pt.all(T0 < T),
            msg="r > 0, alpha > 0, s > 0, beta > 0, T0 < T",
        )
