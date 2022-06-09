import aesara.tensor as at
import numpy as np
from aesara.tensor.random.op import RandomVariable
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters

import pymc as pm


__all__ = [
    "ContNonContract",
    "BetaGeoFitter"
]


class ContNonContractRV(RandomVariable):
    name = "continuous_non_contractual"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ContNonContract", "\\operatorname{ContNonContract}")

    def make_node(self, rng, size, dtype, lam, p, T, T0):

        T = at.as_tensor_variable(T)
        T0 = at.as_tensor_variable(T0)

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

        if size is None:
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

            return np.array(
                [
                    t,
                    n,
                ],
            )

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index], T0[index])

        return output

    def _supp_shape_from_params(*args, **kwargs):
        return (2,)


continuous_non_contractual = ContNonContractRV()


class ContNonContract(PositiveContinuous):
    r"""
    Individual-level model for the customer lifetime value. See equation (3)
    from Fader et al. (2005) [1].

    .. math:

        f(\lambda, p | x, t_1, \dots, t_x, T)
        = f(\lambda, p | t_x, T) = (1 - p)^x \lambda^x \exp(-\lambda T)
          + \delta_{x > 0} p (1 - p)^{x-1} \lambda^x \exp(-\lambda t_x)

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p] = \frac{1}{p}`
                    `- \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`

    References
    ----------
    .. [1] Fader, Peter S., Bruce GS Hardie, and Ka Lok Lee. "“Counting your customers”
           the easy way: An alternative to the Pareto/NBD model." Marketing science
           24.2 (2005): 275-284.
    """
    rv_op = continuous_non_contractual

    @classmethod
    def dist(cls, lam, p, T, T0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

    def get_moment(rv, size, lam, p, T, T0):
        return at.full(size, at.as_tensor_variable([lam * (T - T0), 1 /  p]))

    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]

        zero_observations = at.eq(x, 0)

        A = x * at.log(1 - p) + x * at.log(lam) - lam * (T - T0)
        B = at.log(p) + (x - 1) * at.log(1 - p) + x * at.log(lam) - lam * (t_x - T0)

        logp = at.switch(
            zero_observations,
            A,
            at.logaddexp(A, B),
        )
        logp = at.switch(
            at.any(
                (
                    at.lt(t_x, T0), 
                    at.lt(x, 0),
                    at.gt(t_x, T),
                )
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            lam > 0,
            0 <= p,
            p <= 1,
            at.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


def BetaGeoFitter(name, a, b, r, alpha, T, T0, size=None, shape=None, **kwargs):
    p = pm.Beta(f"{name}_beta", a, b, size=size, shape=shape)
    lam = pm.Gamma(f"{name}_gamma", r, 1/alpha, size=size, shape=shape)
    return ContNonContract(name, lam, p, T, T0, size=size, shape=shape, **kwargs)
