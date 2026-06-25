#   Copyright 2022 - 2026 The PyMC Labs Developers
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

import numpy as np
import pytensor.tensor as pt
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import betaln, check_parameters
from pymc.distributions.distribution import Discrete
from pymc.distributions.shape_utils import rv_size_is_none
from pytensor import scan
from pytensor.graph import vectorize_graph
from pytensor.tensor.random.op import RandomVariable

__all__ = [
    "BetaDiscreteWeibull",
    "BetaGeoBetaBinom",
    "BetaGeoNBD",
    "DiscreteWeibull",
    "ModifiedBetaGeoNBD",
    "ParetoNBD",
]


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

        unnorm_logp = scan(
            fn=logp_customer_died,
            outputs_info=[None],
            sequences=[t_x, x, alpha, beta, gamma, delta, T],
            return_updates=False,
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


class ShiftedBetaGeometricRV(RandomVariable):
    name = "sbg"
    signature = "(),()->()"

    dtype = "int64"
    _print_name = ("ShiftedBetaGeometric", "\\operatorname{ShiftedBetaGeometric}")

    @classmethod
    def rng_fn(cls, rng, alpha, beta, size):
        if size is None:
            size = np.broadcast_shapes(alpha.shape, beta.shape)

        alpha = np.broadcast_to(alpha, size)
        beta = np.broadcast_to(beta, size)

        p_samples = rng.beta(a=alpha, b=beta, size=size)

        # prevent log(0) by clipping small p samples
        p = np.clip(p_samples, 1e-100, 1)
        # TODO: Consider returning np.float64 types instead of np.int64
        #       See relevant PR comment: https://github.com/pymc-labs/pymc-marketing/pull/2010#discussion_r2444116986
        return rng.geometric(p, size=size)


sbg = ShiftedBetaGeometricRV()


class ShiftedBetaGeometric(Discrete):
    r"""Shifted Beta-Geometric distribution.

    This mixture distribution extends the Geometric distribution to support heterogeneity across observations.

    Hardie and Fader describe this distribution with the following PMF and survival functions in [1]_:

    .. math::
        \mathbb{P}(T=t|\alpha,\beta) = \frac{B(\alpha+1,\beta+t-1)}{B(\alpha,\beta)},t=1,2,...  \\
        \begin{align}
        \mathbb{S}(t|\alpha,\beta) = \frac{B(\alpha,\beta+t)}{B(\alpha,\beta)},t=1,2,... \\
        \end{align}

    ========  ===============================================
    Support   :math:`t \in \mathbb{N}_{>0}`
    ========  ===============================================

    Parameters
    ----------
    alpha : tensor_like of float
        Scale parameter (alpha > 0).
    beta : tensor_like of float
        Scale parameter (beta > 0).

    References
    ----------
    .. [1] Fader, P. S., & Hardie, B. G. (2007). How to project customer retention.
           Journal of Interactive Marketing, 21(1), 76-90.
           https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
    """

    rv_op = sbg

    @classmethod
    def dist(cls, alpha, beta, *args, **kwargs):
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)

        return super().dist([alpha, beta], *args, **kwargs)

    def logp(value, alpha, beta):
        """From Expression (5) on p.6 of Fader & Hardie (2007)."""
        logp = betaln(alpha + 1, beta + value - 1) - betaln(alpha, beta)

        logp = pt.switch(
            pt.or_(
                pt.lt(value, 1),
                pt.or_(
                    alpha <= 0,
                    beta <= 0,
                ),
            ),
            -np.inf,
            logp,
        )

        return check_parameters(
            logp,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def logcdf(value, alpha, beta):
        """Adapted from Expression (6) on p.6 of Fader & Hardie (2007)."""
        # survival function from paper
        logS = (
            pt.gammaln(beta + value)
            - pt.gammaln(beta)
            + pt.gammaln(alpha + beta)
            - pt.gammaln(alpha + beta + value)
        )
        return pt.log1mexp(logS)

    def support_point(rv, size, alpha, beta):
        """Calculate a reasonable starting point for sampling.

        For the Shifted Beta-Geometric distribution, we use a point estimate based on
        the expected value of the mixture components.
        """
        geo_mean = pt.ceil(
            pt.reciprocal(
                alpha / (alpha + beta)  # expected value of the beta distribution
            )  # expected value of the geometric distribution
        )
        if not rv_size_is_none(size):
            geo_mean = pt.full(size, geo_mean)
        return geo_mean


# ---------------------------------------------------------------------------
# Discrete-Weibull and Beta-discrete-Weibull (BdW) duration distributions.
# The BdW is the duration-dependence generalization of ShiftedBetaGeometric
# (Fader, Hardie, Liu, Davin & Steenburgh, 2018).
# ---------------------------------------------------------------------------


class DiscreteWeibullRV(RandomVariable):
    """Random variable op for the discrete Weibull (Nakagawa & Osaki)."""

    name = "discrete_weibull"
    signature = "(),()->()"
    dtype = "int64"
    _print_name = ("DiscreteWeibull", "\\operatorname{DiscreteWeibull}")

    @classmethod
    def rng_fn(cls, rng, theta, c, size):
        if size is None:
            size = np.broadcast_shapes(theta.shape, c.shape)
        theta = np.broadcast_to(theta, size).astype(np.float64)
        c = np.broadcast_to(c, size).astype(np.float64)

        # Clip to avoid log(0).  theta is the *churn* probability so
        # theta in (0, 1); we additionally clip c away from zero.
        theta = np.clip(theta, 1e-12, 1.0 - 1e-12)
        c_safe = np.clip(c, 1e-6, None)

        # Inverse-CDF sampling.  For U ~ Uniform(0, 1),
        #   P(T <= t) = 1 - (1 - theta)^(t^c)  >=  U
        # ⇔ t >= (log(1-U) / log(1-theta)) ** (1/c)
        u = rng.uniform(size=size)
        # Clip U to avoid log(0) at the boundary.
        u = np.clip(u, 1e-300, 1 - 1e-300)
        raw = np.log1p(-u) / np.log1p(-theta)  # log(1-U)/log(1-theta)
        t = np.ceil(raw ** (1.0 / c_safe))
        # ``ceil`` of a non-negative real always yields an integer; we
        # enforce a minimum support of 1 (the distribution is shifted).
        t = np.maximum(t, 1).astype(np.int64)
        return t


discrete_weibull = DiscreteWeibullRV()


class DiscreteWeibull(Discrete):
    r"""Discrete Weibull distribution (Nakagawa & Osaki, 1975).

    A shifted discrete lifetime distribution with support
    :math:`T \in \{1, 2, 3, \dots\}` whose survivor function is

    .. math::

        S(t\mid\theta, c) = (1 - \theta)^{t^{c}}, \qquad
        0 < \theta < 1,\; c > 0,\; t = 0, 1, 2, \dots

    and whose probability mass function is

    .. math::

        \mathbb{P}(T = t\mid\theta, c) =
            (1 - \theta)^{(t-1)^{c}} - (1 - \theta)^{t^{c}},
        \qquad t = 1, 2, 3, \dots

    When :math:`c = 1` the distribution reduces to a
    :class:`Geometric(\theta)` — i.e., a constant individual-level hazard.
    For :math:`c > 1` the individual-level churn probability *increases*
    with tenure (positive duration dependence), whereas for :math:`c < 1`
    it *decreases* with tenure (negative duration dependence).

    The individual-level retention probability at period ``t``
    (the probability that a subscriber who has completed ``t-1`` renewals
    renews at the next opportunity) is

    .. math::

        \rho(t\mid\theta, c) = (1 - \theta)^{t^{c} - (t-1)^{c}}.

    ========  ===============================================
    Support   :math:`t \in \mathbb{N}_{>0}`
    ========  ===============================================

    Parameters
    ----------
    theta : tensor_like of float
        Churn probability (0 < theta < 1).
    c : tensor_like of float
        Shape / duration-dependence parameter (c > 0).  ``c == 1`` yields
        the geometric distribution; ``c > 1`` gives increasing churn
        propensity with tenure; ``c < 1`` gives decreasing churn
        propensity with tenure.

    References
    ----------
    .. [1] Nakagawa, T., & Osaki, S. (1975).  The Discrete Weibull
       Distribution.  *IEEE Transactions on Reliability*, 24 (5), 300-301.
    .. [2] Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J., &
       Steenburgh, T. (2018).  "How to Project Customer Retention"
       Revisited: The Role of Duration Dependence.
       https://brucehardie.com/papers/037/
    """

    rv_op = discrete_weibull

    @classmethod
    def dist(cls, theta, c, *args, **kwargs):
        """Create a DiscreteWeibull distribution."""
        theta = pt.as_tensor_variable(theta)
        c = pt.as_tensor_variable(c)
        return super().dist([theta, c], *args, **kwargs)

    def logp(value, theta, c):
        """Log-pmf of the discrete Weibull distribution.

        Adapted from equation (8) of Fader et al. (2018).  Implemented in
        log-space using ``log1mexp`` for numerical stability.
        """
        t = pt.cast(value, "floatX")
        # (t-1)^c  with the convention 0^c = 0 for c > 0 and a safe zero
        # when t == 1.  Using ``switch`` avoids pytensor evaluating 0**c
        # for every element of the sample.
        t_minus_1_c = pt.switch(pt.gt(t, 1), pt.power(pt.maximum(t - 1, 0.0), c), 0.0)
        t_c = pt.power(t, c)

        # log(1 - theta) computed as log1p(-theta) for numerical stability.
        log_q = pt.log1p(-theta)

        # logS(t-1) and logS(t) where S(t) = (1 - theta)^{t^c}.
        log_S_t_minus_1 = log_q * t_minus_1_c
        log_S_t = log_q * t_c

        # log P(T=t) = log(S(t-1) - S(t))
        #            = log_S_t_minus_1 + log(1 - exp(log_S_t - log_S_t_minus_1))
        #
        # ``pt.log1mexp(x)`` returns ``log(1 - exp(x))`` and expects x <= 0.
        # We have log_S_t <= log_S_t_minus_1 so the argument is non-positive.
        logp = log_S_t_minus_1 + pt.log1mexp(log_S_t - log_S_t_minus_1)

        logp = pt.switch(
            pt.or_(pt.lt(value, 1), pt.or_(theta <= 0, theta >= 1)),
            -np.inf,
            logp,
        )
        return check_parameters(
            logp,
            theta > 0,
            theta < 1,
            c > 0,
            msg="0 < theta < 1, c > 0",
        )

    def logcdf(value, theta, c):
        """Log-cdf of the discrete Weibull distribution.

        ``log CDF(t) = log(1 - S(t)) = log(1 - (1 - theta)^{t^c})``.
        """
        t = pt.cast(value, "floatX")
        t_c = pt.power(t, c)
        log_q = pt.log1p(-theta)
        logS = log_q * t_c
        return pt.log1mexp(logS)

    def support_point(rv, size, theta, c):
        """Return a reasonable starting point for samplers.

        We use the median of the continuous Weibull with the equivalent
        scale/shape (``λ = -log(1 - theta)``) rounded up to the next
        integer.  This gives a smooth mode-like value for small ``theta``
        without special-casing the discrete support.
        """
        log_q = pt.log1p(-theta)
        median = pt.ceil(pt.power(pt.log(2.0) / (-log_q), pt.reciprocal(c)))
        median = pt.maximum(median, 1)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median


# ---------------------------------------------------------------------------
# Beta-discrete-Weibull — population-level duration distribution obtained by
# mixing the DiscreteWeibull over a Beta(alpha, beta) heterogeneity
# distribution on ``theta``.  This is the direct extension of
# ``ShiftedBetaGeometric`` described by Fader et al. (2018).
# ---------------------------------------------------------------------------


class BetaDiscreteWeibullRV(RandomVariable):
    """Random variable op for the Beta-discrete-Weibull (BdW) distribution."""

    name = "beta_discrete_weibull"
    signature = "(),(),()->()"
    dtype = "int64"
    _print_name = ("BetaDiscreteWeibull", "\\operatorname{BetaDiscreteWeibull}")

    @classmethod
    def rng_fn(cls, rng, alpha, beta, c, size):
        if size is None:
            size = np.broadcast_shapes(alpha.shape, beta.shape, c.shape)
        alpha = np.broadcast_to(alpha, size)
        beta = np.broadcast_to(beta, size)
        c = np.broadcast_to(c, size).astype(np.float64)

        # Draw theta ~ Beta(alpha, beta) per observation, then sample a
        # discrete Weibull with parameters (theta, c) via inverse CDF.
        theta = rng.beta(a=alpha, b=beta, size=size)
        theta = np.clip(theta, 1e-12, 1.0 - 1e-12)
        c_safe = np.clip(c, 1e-6, None)

        u = rng.uniform(size=size)
        u = np.clip(u, 1e-300, 1 - 1e-300)

        raw = np.log1p(-u) / np.log1p(-theta)
        t = np.ceil(raw ** (1.0 / c_safe))
        t = np.maximum(t, 1).astype(np.int64)
        return t


beta_discrete_weibull = BetaDiscreteWeibullRV()


class BetaDiscreteWeibull(Discrete):
    r"""Beta-discrete-Weibull distribution (Fader et al., 2018).

    Population-level duration distribution for contractual customer
    relationships.  Obtained by compounding a discrete Weibull lifetime
    with a Beta-distributed heterogeneity on the churn probability:

    .. math::

        \theta \sim \mathrm{Beta}(\alpha, \beta), \qquad
        T \mid \theta, c \sim \mathrm{DiscreteWeibull}(\theta, c).

    The resulting survivor and probability-mass functions are

    .. math::

        S(t \mid \alpha, \beta, c) &=
            \frac{B(\alpha, \beta + t^{c})}{B(\alpha, \beta)}, \\[4pt]
        \mathbb{P}(T = t \mid \alpha, \beta, c) &=
            \frac{B(\alpha, \beta + (t-1)^{c}) -
                  B(\alpha, \beta + t^{c})}{B(\alpha, \beta)},
        \qquad t = 1, 2, 3, \dots

    When :math:`c = 1` the BdW collapses to the shifted beta-geometric
    (sBG) distribution of Fader & Hardie (2007) — the two families share
    the same :math:`(\alpha, \beta)` parameterisation used in
    pymc-marketing.

    Cohort-level retention rate (equation 12 of [1]_):

    .. math::

        r(t \mid \alpha, \beta, c) =
            \frac{S(t)}{S(t-1)}
            = \frac{B(\alpha, \beta + t^{c})}{B(\alpha, \beta + (t-1)^{c})}.

    ========  ===============================================
    Support   :math:`t \in \mathbb{N}_{>0}`
    ========  ===============================================

    Parameters
    ----------
    alpha : tensor_like of float
        Shape parameter of the Beta mixing distribution (``alpha > 0``).
    beta : tensor_like of float
        Shape parameter of the Beta mixing distribution (``beta > 0``).
    c : tensor_like of float
        Duration-dependence shape parameter (``c > 0``).  ``c == 1`` gives
        the shifted-beta-geometric (sBG) model; ``c > 1`` corresponds to
        increasing individual-level churn propensity with tenure; ``c < 1``
        corresponds to decreasing individual-level churn propensity.

    References
    ----------
    .. [1] Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J.,
       & Steenburgh, T. (2018).  "How to Project Customer Retention"
       Revisited: The Role of Duration Dependence.
       https://brucehardie.com/papers/037/
    .. [2] Fader, P. S., & Hardie, B. G. S. (2007).  How to project
       customer retention.  *Journal of Interactive Marketing*, 21 (1),
       76-90.
    """

    rv_op = beta_discrete_weibull

    @classmethod
    def dist(cls, alpha, beta, c, *args, **kwargs):
        """Create a BetaDiscreteWeibull distribution."""
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)
        c = pt.as_tensor_variable(c)
        return super().dist([alpha, beta, c], *args, **kwargs)

    def logp(value, alpha, beta, c):
        """Log-pmf of the BdW, equation (11) of Fader et al. (2018).

        Numerically stable computation using log-space differences and
        ``log1mexp``.
        """
        t = pt.cast(value, "floatX")
        t_c = pt.power(t, c)
        t_minus_1_c = pt.switch(pt.gt(t, 1), pt.power(pt.maximum(t - 1, 0.0), c), 0.0)

        log_S_t_minus_1 = betaln(alpha, beta + t_minus_1_c) - betaln(alpha, beta)
        log_S_t = betaln(alpha, beta + t_c) - betaln(alpha, beta)

        # log P(T=t) = log(S(t-1) - S(t))
        logp = log_S_t_minus_1 + pt.log1mexp(log_S_t - log_S_t_minus_1)

        logp = pt.switch(
            pt.or_(
                pt.lt(value, 1),
                pt.or_(alpha <= 0, pt.or_(beta <= 0, c <= 0)),
            ),
            -np.inf,
            logp,
        )
        return check_parameters(
            logp,
            alpha > 0,
            beta > 0,
            c > 0,
            msg="alpha > 0, beta > 0, c > 0",
        )

    def logcdf(value, alpha, beta, c):
        """Log-cdf of the BdW.

        ``log CDF(t) = log(1 - S(t))`` with
        ``log S(t) = log B(alpha, beta + t^c) - log B(alpha, beta)``.
        """
        t = pt.cast(value, "floatX")
        t_c = pt.power(t, c)
        logS = betaln(alpha, beta + t_c) - betaln(alpha, beta)
        return pt.log1mexp(logS)

    def support_point(rv, size, alpha, beta, c):
        r"""Return a starting point for samplers.

        Uses the same heuristic as :class:`ShiftedBetaGeometric` (the
        reciprocal of the prior mean of :math:`\theta`) since the BdW
        reduces to the sBG at ``c = 1`` and the sBG starting point is
        already a sensible default across the parameter space.
        """
        geo_mean = pt.ceil(pt.reciprocal(alpha / (alpha + beta)))
        if not rv_size_is_none(size):
            geo_mean = pt.full(size, geo_mean)
        return geo_mean
