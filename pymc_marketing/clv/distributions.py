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
"""Distributions for the CLV module."""

from functools import reduce

import numpy as np
import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import betaln, check_parameters
from pymc.distributions.distribution import Discrete
from pytensor import scan
from pytensor.graph import vectorize_graph
from pytensor.tensor.random.op import RandomVariable

__all__ = [
    "BetaGeoBetaBinom",
    "BetaGeoNBD",
    "ContContract",
    "ContNonContract",
    "ModifiedBetaGeoNBD",
    "ParetoNBD",
]


class ContNonContractRV(RandomVariable):
    name = "continuous_non_contractual"
    signature = "(),(),()->(2)"
    dtype = "floatX"
    _print_name = ("ContNonContract", "\\operatorname{ContNonContract}")

    def __call__(self, lam, p, T, size=None, **kwargs):
        return super().__call__(lam, p, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, size):
        if size is None:
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


continuous_non_contractual = ContNonContractRV()


class ContNonContract(PositiveContinuous):
    r"""Individual-level model for the customer lifetime value.

    See equation (3) from Fader et al. (2005) [1]_.

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
        """Get the distribution from the parameters."""
        return super().dist([lam, p, T], **kwargs)

    def logp(value, lam, p, T):
        """Log-likelihood of the distribution."""
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
            reduce(
                pt.bitwise_or,
                [
                    pt.and_(pt.ge(t_x, 0), zero_observations),
                    pt.lt(t_x, 0),
                    pt.lt(x, 0),
                    pt.gt(t_x, T),
                ],
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
    signature = "(),(),()->(3)"
    dtype = "floatX"
    _print_name = ("ContinuousContractual", "\\operatorname{ContinuousContractual}")

    def __call__(self, lam, p, T, size=None, **kwargs):
        return super().__call__(lam, p, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, size):
        if size is None:
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
    r"""Distribution class of a continuous contractual data-generating process.

    That is where purchases can occur at any time point (continuous) and churning/dropping
    out is explicit (contractual).

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
        """Get the distribution from the parameters."""
        return super().dist([lam, p, T], **kwargs)

    def logp(value, lam, p, T):
        """Log-likelihood of the distribution."""
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
            reduce(
                pt.bitwise_or,
                [
                    zero_observations,
                    pt.lt(t_x, 0),
                    pt.lt(x, 0),
                    pt.gt(t_x, T),
                    pt.bitwise_not(pt.bitwise_or(pt.eq(churn, 0), pt.eq(churn, 1))),
                ],
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
    signature = "(),(),(),(),()->(2)"
    dtype = "floatX"
    _print_name = ("ParetoNBD", "\\operatorname{ParetoNBD}")

    def __call__(self, r, alpha, s, beta, T, size=None, **kwargs):
        return super().__call__(r, alpha, s, beta, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, r, alpha, s, beta, T, size):
        if size is None:
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


pareto_nbd = ParetoNBDRV()


class ParetoNBD(PositiveContinuous):
    r"""Population-level distribution class for a continuous, non-contractual, Pareto/NBD process.

    It is based on Schmittlein, et al. in [2]_.

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
    Support   :math:`t_j >= 0` for :math:`j = 1, \dots, x`
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
        """Get the distribution from the parameters."""
        return super().dist([r, alpha, s, beta, T], **kwargs)

    def logp(value, r, alpha, s, beta, T):
        """Log-likelihood of the distribution."""
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


class BetaGeoBetaBinomRV(RandomVariable):
    name = "beta_geo_beta_binom"
    signature = "(),(),(),(),()->(2)"
    dtype = "floatX"
    _print_name = ("BetaGeoBetaBinom", "\\operatorname{BetaGeoBetaBinom}")

    def __call__(self, alpha, beta, gamma, delta, T, size=None, **kwargs):
        return super().__call__(alpha, beta, gamma, delta, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, alpha, beta, gamma, delta, T, size) -> np.ndarray:
        if size is None:
            size = np.broadcast_shapes(
                alpha.shape, beta.shape, gamma.shape, delta.shape, T.shape
            )

        alpha = np.broadcast_to(alpha, size)
        beta = np.broadcast_to(beta, size)
        gamma = np.broadcast_to(gamma, size)
        delta = np.broadcast_to(delta, size)
        T = np.broadcast_to(T, size)

        output = np.zeros(shape=(*size, 2))

        purchase_prob = rng.beta(a=alpha, b=beta, size=size)
        churn_prob = rng.beta(a=delta, b=gamma, size=size)

        def sim_data(purchase_prob, churn_prob, T):
            t_x = 0
            x = 0
            active = True
            recency = 0

            while t_x <= T and active:
                t_x += 1
                active = rng.binomial(1, churn_prob)
                purchase = rng.binomial(1, purchase_prob)
                if active and purchase:
                    recency = t_x
                    x += 1
            return np.array(
                [
                    recency if x > 0 else T,
                    x,
                ],
            )

        for index in np.ndindex(*size):
            output[index] = sim_data(purchase_prob[index], churn_prob[index], T[index])

        return output


beta_geo_beta_binom = BetaGeoBetaBinomRV()


class BetaGeoBetaBinom(Discrete):
    r"""Population-level distribution class for a discrete, non-contractual, Beta-Geometric/Beta-Binomial process.

    It is based on equation(5) from Fader, et al. in [1]_.

    .. math::

        \mathbb{L}(\alpha, \beta, \gamma, \delta  | x, t_x, n) &=
        \frac{B(\alpha+x,\beta+n-x)}{B(\alpha,\beta)}
        \frac{B(\gamma,\delta+n)}{B(\gamma,\delta)} \\
        &+ \sum_{i=0}^{n-t_x-1}\frac{B(\alpha+x,\beta+t_x-x+i)}{B(\alpha,\beta)} \\
        &\cdot \frac{B(\gamma+1,\delta+t_x+i)}{B(\gamma,\delta)}

    ========  ===============================================
    Support   :math:`t_j >= 0` for :math:`j = 1, \dots,x`
    Mean      :math:`\mathbb{E}[X(n) | \alpha, \beta, \gamma, \delta] =  (\frac{\alpha}{\alpha+\beta})(\frac{\delta}{\gamma-1}) \cdot{1-\frac{\Gamma(\gamma+\delta)}{\Gamma(\gamma+\delta+n)}\frac{\Gamma(1+\delta+n)}{\Gamma(1+ \delta)}}`
    ========  ===============================================

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Customer-Base Analysis in a Discrete-Time Noncontractual Setting,"
       Marketing Science, 29 (6), 1086-1108. https://www.brucehardie.com/papers/020/fader_et_al_mksc_10.pdf

    """  # noqa: E501

    rv_op = beta_geo_beta_binom

    @classmethod
    def dist(cls, alpha, beta, gamma, delta, T, **kwargs):
        """Get the distribution from the parameters."""
        return super().dist([alpha, beta, gamma, delta, T], **kwargs)

    def logp(value, alpha, beta, gamma, delta, T):
        """Log-likelihood of the distribution."""
        t_x = pt.atleast_1d(value[..., 0])
        x = pt.atleast_1d(value[..., 1])

        for param in (t_x, x, alpha, beta, gamma, delta, T):
            if param.type.ndim > 1:
                raise NotImplementedError(
                    f"BetaGeoBetaBinom logp only implemented for vector parameters, got ndim={param.type.ndim}"
                )

        # Broadcast all the parameters so they are sequences.
        # Potentially inefficient, but otherwise ugly logic needed to unpack arguments in the scan function,
        # since sequences always precede non-sequences.
        t_x, alpha, beta, gamma, delta, T = pt.broadcast_arrays(
            t_x, alpha, beta, gamma, delta, T
        )

        def logp_customer_died(t_x_i, x_i, alpha_i, beta_i, gamma_i, delta_i, T_i):
            i = pt.scalar("i", dtype=int)
            died = pt.lt(t_x_i + i, T_i)

            unnorm_logprob_customer_died_at_tx_plus_i = betaln(
                alpha_i + x_i, beta_i + t_x_i - x_i + i
            ) + betaln(gamma_i + died, delta_i + t_x_i + i)

            # Maximum prevents invalid T - t_x values from crashing logp
            i_vec = pt.arange(pt.maximum(T_i - t_x_i, 0) + 1)
            unnorm_logprob_customer_died_at_tx_plus_i_vec = vectorize_graph(
                unnorm_logprob_customer_died_at_tx_plus_i, replace={i: i_vec}
            )

            return pt.logsumexp(unnorm_logprob_customer_died_at_tx_plus_i_vec)

        unnorm_logp, _ = scan(
            fn=logp_customer_died,
            outputs_info=[None],
            sequences=[t_x, x, alpha, beta, gamma, delta, T],
        )

        logp = unnorm_logp - betaln(alpha, beta) - betaln(gamma, delta)

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

        if value.ndim == 1:
            logp = pt.specify_shape(logp, 1).squeeze(0)

        return check_parameters(
            logp,
            alpha > 0,
            beta > 0,
            gamma > 0,
            delta > 0,
            msg="alpha > 0, beta > 0, gamma > 0, delta > 0",
        )


class BetaGeoNBDRV(RandomVariable):
    name = "bg_nbd"
    signature = "(),(),(),(),()->(2)"

    dtype = "floatX"
    _print_name = ("BetaGeoNBD", "\\operatorname{BetaGeoNBD}")

    def __call__(self, a, b, r, alpha, T, size=None, **kwargs):
        return super().__call__(a, b, r, alpha, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, a, b, r, alpha, T, size):
        if size is None:
            size = np.broadcast_shapes(a.shape, b.shape, r.shape, alpha.shape, T.shape)

        a = np.asarray(a)
        b = np.asarray(b)
        r = np.asarray(r)
        alpha = np.asarray(alpha)
        T = np.asarray(T)

        if size == ():
            size = np.broadcast_shapes(a.shape, b.shape, r.shape, alpha.shape, T.shape)

        a = np.broadcast_to(a, size)
        b = np.broadcast_to(b, size)
        r = np.broadcast_to(r, size)
        alpha = np.broadcast_to(alpha, size)
        T = np.broadcast_to(T, size)

        output = np.zeros(shape=size + (2,))  # noqa:RUF005

        lam = rng.gamma(shape=r, scale=1 / alpha, size=size)
        p = rng.beta(a=a, b=b, size=size)

        def sim_data(lam, p, T):
            t = 0  # recency
            n = 0  # frequency

            churn = 0  # BG/NBD assumes all non-repeat customers are active
            wait = rng.exponential(scale=1 / lam)

            while t + wait < T and not churn:
                churn = rng.random() < p
                n += 1
                t += wait
                wait = rng.exponential(scale=1 / lam)

            return np.array([t, n])

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index])

        return output


bg_nbd = BetaGeoNBDRV()


class BetaGeoNBD(PositiveContinuous):
    r"""Population-level distribution class for a discrete, non-contractual, Beta-Geometric/Negative-Binomial process.

    Based on Fader, et al. in [1]_, [2]_ and enhancements for numerical stability in [3]_.

    .. math::

        \mathbb{LL}(r, \alpha, a, b  | x, t_x, T) =
        D_1 + D_2 + \ln(C_3 + \delta_{x>0} C_4) \text{, where:} \\
        \begin{align}
        D_1 &= \ln \left[ \Gamma(r+x) \right] - \ln \left[ \Gamma(r) \right] + \ln \left[ \Gamma(a+b) \right] + \ln \left[ \Gamma(b+x) \right] \\
        D_2 &= r \ln(\alpha) - (r+x) \ln(\alpha + t_x) \\
        C_3 &= \left(\frac{\alpha + t_x}{\alpha + T} \right)^{r+x} \\
        C_4 &= \left(\frac{a}{b+x-1} \right) \\
        \end{align}

    ========  ===============================================
    Support   :math:`t_j >= 0` for :math:`j = 1, \dots,x`
    Mean      :math:`\mathbb{E}[X(n) | r, \alpha, a, b] = \frac{a+b-1}{a-1} \left[ 1 - \left(\frac{\alpha}{\alpha + T}\right)^r {_2}{F}{_1}(r,b;a+b-1;\frac{t}{\alpha + t}) \right]`
    ========  ===============================================

    References
    ----------
    .. [1] Fader, Peter S., Bruce G.S. Hardie, and Jen Shang (2010),
       "Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model
       Marketing Science, 24 (Spring), 275-284

    .. [2] Implementing the BG/NBD Model for Customer Base Analysis in Excel http://brucehardie.com/notes/004/bgnbd_spreadsheet_note.pdf
    .. [3] Overcoming the BG/NBD Model's #NUM! Error Problem https://brucehardie.com/notes/027/bgnbd_num_error.pdf

    """  # noqa: E501

    rv_op = bg_nbd

    @classmethod
    def dist(cls, a, b, r, alpha, T, **kwargs):
        """Get the distribution from the parameters."""
        return super().dist([a, b, r, alpha, T], **kwargs)

    def logp(value, a, b, r, alpha, T):
        """Log-likelihood of the distribution."""
        t_x = pt.atleast_1d(value[..., 0])
        x = pt.atleast_1d(value[..., 1])

        for param in (t_x, x, a, b, r, alpha, T):
            if param.type.ndim > 1:
                raise NotImplementedError(
                    f"BetaGeoNBD logp only implemented for vector parameters, got ndim={param.type.ndim}"
                )

        x_non_zero = x > 0

        d1 = (
            pt.gammaln(r + x)
            - pt.gammaln(r)
            + pt.gammaln(a + b)
            + pt.gammaln(b + x)
            - pt.gammaln(b)
            - pt.gammaln(a + b + x)
        )

        d2 = r * pt.log(alpha) - (r + x) * pt.log(alpha + t_x)
        c3 = ((alpha + t_x) / (alpha + T)) ** (r + x)
        c4 = a / (b + x - 1)

        logp = d1 + d2 + pt.log(c3 + pt.switch(x_non_zero, c4, 0))

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
            a > 0,
            b > 0,
            alpha > 0,
            r > 0,
            msg="a > 0, b > 0, alpha > 0, r > 0",
        )


class ModifiedBetaGeoNBDRV(RandomVariable):
    name = "mbg_nbd"
    signature = "(),(),(),(),()->(2)"

    dtype = "floatX"
    _print_name = ("ModifiedBetaGeoNBD", "\\operatorname{ModifiedBetaGeoNBD}")

    def __call__(self, a, b, r, alpha, T, size=None, **kwargs):
        return super().__call__(a, b, r, alpha, T, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, a, b, r, alpha, T, size):
        if size is None:
            size = np.broadcast_shapes(a.shape, b.shape, r.shape, alpha.shape, T.shape)

        a = np.asarray(a)
        b = np.asarray(b)
        r = np.asarray(r)
        alpha = np.asarray(alpha)
        T = np.asarray(T)

        if size == ():
            size = np.broadcast_shapes(a.shape, b.shape, r.shape, alpha.shape, T.shape)

        a = np.broadcast_to(a, size)
        b = np.broadcast_to(b, size)
        r = np.broadcast_to(r, size)
        alpha = np.broadcast_to(alpha, size)
        T = np.broadcast_to(T, size)

        output = np.zeros(shape=size + (2,))  # noqa:RUF005

        lam = rng.gamma(shape=r, scale=1 / alpha, size=size)
        p = rng.beta(a=a, b=b, size=size)

        def sim_data(lam, p, T):
            t = 0  # recency
            n = 0  # frequency

            churn = (
                rng.random() < p
            )  # MBG/NBD customer active with probability p at time 0
            wait = rng.exponential(scale=1 / lam)

            while t + wait < T and not churn:
                churn = rng.random() < p
                n += 1
                t += wait
                wait = rng.exponential(scale=1 / lam)

            return np.array([t, n])

        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index])

        return output


mbg_nbd = ModifiedBetaGeoNBDRV()


class ModifiedBetaGeoNBD(PositiveContinuous):
    r"""Population-level distribution for a discrete, non-contractual Modified-Beta-Geometric/Negative-Binomial process.

    In MBG/NBD, a customer may drop out at time zero. This is in contrast with the BG/NBD model,
    which assumes all non-repeat customers are still active.
    Based on Batislam, et al. in [1]_, and Wagner & Hopper in [2]_ .

    .. math::
        \mathbb{LL}(a, b, \alpha, r | x, t_x, T) = \ln \left[
        A_1 * A_2 * (A_3 + \delta_{x>0} A_4) \right] \text{, where:} \\
        \begin{align}
        A_1 &= \frac{\Gamma(r+x) \alpha^r)}{\Gamma(x)} \\
        A_2 &= \frac{\Gamma(a+b) \Gamma(b+x+1)}{\Gamma(b) \Gamma(a+b+x+1)} \\
        A_3 &= \left( \frac{1}{\alpha + T} \right)^(r+x) \\
        A_4 &= \left( \frac{a}{b+x} \right) \left( \frac{1}{\alpha + t_x} \right)^(r+x) \\
        \end{align}
    ========  ===============================================
    Support   :math:`t_j >= 0` for :math:`j = 1, \dots,x`
    Mean      :math:`\mathbb{E}[X(n) | r, \alpha, a, b] = \frac{a+b-1}{a-1} \left[ 1 - \left(\frac{\alpha}{\alpha + T}\right)^r {_2}{F}{_1}(r,b;a+b-1;\frac{t}{\alpha + t}) \right]`
    ========  ===============================================

    References
    ----------
    .. [1] Batislam, E.P., M. Denizel, A. Filiztekin (2007),
       "Empirical validation and comparison of models for customer base
       analysis,"
       International Journal of Research in Marketing, 24 (3), 201-209.
    .. [2] Wagner, U. and Hoppe D. (2008), "Erratum on the MBG/NBD Model,"
       International Journal of Research in Marketing, 25 (3), 225-226.
    """  # noqa: E501

    rv_op = mbg_nbd

    @classmethod
    def dist(cls, a, b, r, alpha, T, **kwargs):
        """Get the distribution from the parameters."""
        return super().dist([a, b, r, alpha, T], **kwargs)

    def logp(value, a, b, r, alpha, T):
        """Log-likelihood of the distribution."""
        t_x = pt.atleast_1d(value[..., 0])
        x = pt.atleast_1d(value[..., 1])

        for param in (t_x, x, a, b, r, alpha, T):
            if param.type.ndim > 1:
                raise NotImplementedError(
                    f"ModifiedBetaGeoNBD logp only implemented for vector parameters, got ndim={param.type.ndim}"
                )

        a1 = pt.gammaln(r + x) - pt.gammaln(r) + r * pt.log(alpha)
        a2 = (
            pt.gammaln(a + b)
            + pt.gammaln(b + x + 1)
            - pt.gammaln(b)
            - pt.gammaln(a + b + x + 1)
        )
        a3 = -(r + x) * pt.log(alpha + T)
        a4 = (
            pt.log(a)
            - pt.log(b + x)
            + (r + x) * (pt.log(alpha + T) - pt.log(alpha + t_x))
        )

        logp = a1 + a2 + a3 + pt.logaddexp(a4, 0)

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
            a > 0,
            b > 0,
            alpha > 0,
            r > 0,
            msg="a > 0, b > 0, alpha > 0, r > 0",
        )
