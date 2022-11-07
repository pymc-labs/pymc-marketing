import aesara
import aesara.tensor as at
import numpy as np
import pymc as pm
from aeppl.joint_logprob import _logprob
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.utils import normalize_size_param
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import SymbolicRandomVariable
from pymc.distributions.shape_utils import _change_dist_size

__all__ = [
    "ContNonContract",
    "ContContract",
    "GammaGamma",
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

        # TODO: broadcast sizes
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
    def dist(cls, lam, p, T, T0=0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

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
            at.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


class ContContractRV(RandomVariable):
    name = "continuous_contractual"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("ContinuousContractual", "\\operatorname{ContinuousContractual}")

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
    .. math:

        f(\lambda, p | d, x, t_1, \dots, t_x, T)
        = f(\lambda, p | t_x, T) = (1 - p)^{x-1} \lambda^x \exp(-\lambda t_x)
        p^d \left\{(1-p)\exp(-\lambda*(T - t_x))\right\}^{1 - d}

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p, d] = \frac{1}{p}`
                    `- \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`
    """
    rv_op = continuous_contractual

    @classmethod
    def dist(cls, lam, p, T, T0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)

    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]
        churn = value[..., 2]

        zero_observations = at.eq(x, 0)

        logp = (x - 1) * at.log(1 - p) + x * at.log(lam) - lam * t_x
        logp += churn * at.log(p) + (1 - churn) * (
            at.log(1 - p) - lam * ((T - T0) - t_x)
        )

        logp = at.switch(
            zero_observations,
            -lam * (T - T0),
            logp,
        )

        logp = at.switch(
            at.any(at.or_(at.lt(t_x, 0), at.lt(x, 0))),
            -np.inf,
            logp,
        )
        logp = at.switch(
            at.all(
                at.or_(at.eq(churn, 0), at.eq(churn, 1)),
            ),
            logp,
            -np.inf,
        )
        logp = at.switch(
            at.any(
                (
                    at.lt(t_x, T0),
                    at.lt(x, 0),
                    at.gt(t_x, T),
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
            at.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


class GammaGammaRV(SymbolicRandomVariable):
    """Gamma-Gamma compound random variable"""

    _print_name = ("GammaGamma", "\\operatorname{GammaGamma}")
    default_output = 2

    def update(self, node):
        """Symbolic update expression for input random state variables

        Returns a dictionary with the symbolic expressions required for correct updating
        of random state input variables repeated function evaluations. This is used by
        `aesaraf.compile_pymc`.
        """
        nu_rng, out_rng, *_ = node.inputs
        next_nu_rng, next_out_rng, _ = node.outputs
        return {nu_rng: next_nu_rng, out_rng: next_out_rng}


class GammaGamma(PositiveContinuous):
    r"""
    Distribution class for the gamma-gamma model for modelling customer purchases.
    .. math:

        f(z_1, \dots, z_x)
        = f(\overline{z} | x)
        = \frac{\Gamma(px + q)}{\Gamma(px)\Gamma(q)}
        \frac{v^q \overline{z}^{px - 1}x^{px}}{(x \overline{z} + v)^{px + q}}

    ========  ===============================================
    Support   :math:`t_j > 0` for :math:`j = 1, \dots, x`
    Mean      :math:`\mathbb{E}[X(t) | \lambda, p, d] = \frac{1}{p}`
                    `- \frac{1}{p}\exp\left(-\lambda p \min(t, T)\right)`

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2013). The Gamma-Gamma model of monetary
           value. February, 2, 1-9. http://www.brucehardie.com/notes/025/gamma_gamma.pdf
    """
    rv_type = GammaGammaRV

    @classmethod
    def dist(cls, *, p, q, v, x, **kwargs):
        p = pm.floatX(at.as_tensor_variable(p))
        q = pm.floatX(at.as_tensor_variable(q))
        v = pm.floatX(at.as_tensor_variable(v))
        x = pm.intX(at.as_tensor_variable(x))
        return super().dist([p, q, v, x], **kwargs)

    @classmethod
    def rv_op(cls, p, q, v, x, size=None):
        assert p.ndim == 0
        assert q.ndim == 0
        assert v.ndim == 0

        nu_rng = aesara.shared(np.random.default_rng())
        out_rng = aesara.shared(np.random.default_rng())
        if size is None:
            size = x.shape
        size = normalize_size_param(size)

        # Create symbolic RV graph with dummy variables
        nu_rng_, out_rng_, size_ = nu_rng.type(), out_rng.type(), size.type()
        p_, q_, v_, x_ = p.type(), q.type(), v.type(), x.type()
        next_nu_rng_, nu_ = pm.Gamma.dist(q_, v_, size=size_, rng=nu_rng_).owner.outputs
        next_out_rng_, out_ = pm.Gamma.dist(
            p_ * x_, nu_ * x_, size=size_, rng=out_rng_
        ).owner.outputs

        return GammaGammaRV(
            inputs=[nu_rng_, out_rng_, size_, p_, q_, v_, x_],
            outputs=[next_nu_rng_, next_out_rng_, out_],
            ndim_supp=0,
            inplace=True,
        )(nu_rng, out_rng, size, p, q, v, x)


@_change_dist_size.register(GammaGammaRV)
def change_gamma_gamma_size(op, dist, new_size, expand=False):
    *_, old_size, p, q, v, x = dist.owner.inputs

    if expand:
        new_size = tuple(new_size) + tuple(old_size)

    return GammaGamma.rv_op(p, q, v, x, size=new_size)


@_logprob.register(GammaGammaRV)
def gamma_gamma_logp(op, value, nu_rng, out_rng, size, p, q, v, x, **kwargs):
    # Likelihood for mean_spend, marginalizing over nu
    # Eq 1a from [1], p.2
    return (
        at.gammaln(p * x + q)
        - at.gammaln(p * x)
        - at.gammaln(q)
        + q * at.log(v)
        + (p * x - 1) * at.log(value)
        + (p * x) * at.log(x)
        - (p * x + q) * at.log(x * value + v)
    )
