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
r"""Reweight posterior draws to a changed prior or likelihood with PSIS.

Fit once, then answer "what if the prior (or likelihood) were different?" from
the saved draws. For draws :math:`\theta_s` of the reference posterior, the
posterior under a changed model has importance weights

.. math::

    \log w_s = \sum_i \left[\log p_{new,i}(\theta_s) - \log p_{ref,i}(\theta_s)\right]

over the terms that changed. Terms that did not change cancel and must be left
out. :func:`psis_reweight` smooths these weights with Pareto-smoothed importance
sampling (PSIS, [1]_) and reports the Pareto :math:`\hat{k}` and
the effective sample size (ESS), which say when the reweighted posterior can be
trusted and when to refit.

References
----------
.. [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. (2024).
   Pareto smoothed importance sampling. Journal of Machine Learning Research,
   25(72), 1-58.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import arviz as az
import numpy as np
import numpy.typing as npt
import xarray as xr
from arviz_stats.base import array_stats
from scipy.special import logsumexp

__all__ = ["PSISReweighting", "psis_reweight", "relative_efficiency"]

#: Upper bounds of the Pareto :math:`\hat{k}` ranges "use it" and "check".
K_USE_MAX = 0.5
K_CHECK_MAX = 0.7

#: Default ESS below which the reweighted posterior is too thin to summarize.
MIN_ESS = 100.0

#: Absolute spread of the log ratio below which it counts as constant.
_CONSTANT_ATOL = 1e-10

_VERDICTS = ("use it", "check", "refit")


@dataclass(frozen=True)
class PSISReweighting:
    r"""Smoothed weights of a reweighting and its diagnostics.

    Attributes
    ----------
    weights : numpy.ndarray
        Normalized importance weights, shape ``(n_draws,)``.
    pareto_k : float
        Pareto :math:`\hat{k}` of the fitted tail. ``nan`` when the log ratio is
        constant, because there is nothing to smooth.
    ess : float
        Effective sample size of the weights, ``r_eff / sum(weights**2)``. The
        factor ``r_eff`` accounts for the autocorrelation of the MCMC draws, as
        in the ``loo`` R package and ArviZ's moment matching.
    ess_fraction : float
        ``ess`` as a fraction of the number of draws.
    verdict : {"use it", "check", "refit"}
        ``"use it"`` below :math:`\hat{k}` 0.5, ``"check"`` from 0.5 to 0.7 and
        ``"refit"`` from 0.7 on, or whenever ``ess`` is below the floor passed to
        :func:`psis_reweight`. :math:`\hat{k}` cannot see a pure truncation (it
        is ``nan`` there), so the ESS floor is what catches one that keeps too
        few draws.
    """

    weights: npt.NDArray[np.float64]
    pareto_k: float
    ess: float
    ess_fraction: float
    verdict: str


def psis_reweight(
    log_ratio: npt.ArrayLike, r_eff: float = 1.0, min_ess: float = MIN_ESS
) -> PSISReweighting:
    r"""Pareto-smoothed importance weights of a log importance ratio.

    Parameters
    ----------
    log_ratio : array-like, shape (n_draws,)
        ``log p_new - log p_ref`` per posterior draw, with chains and draws
        merged. Draws with ``-inf`` (zero density under the new model) get
        weight 0 and are left out of the smoothing.
    r_eff : float, default 1.0
        Relative efficiency of the reference draws, the mean ESS over the number
        of draws, see :func:`relative_efficiency`. It sets the length of the
        Pareto tail and scales the reported ESS.
    min_ess : float, default 100.0
        ESS below which the verdict is ``"refit"`` whatever :math:`\hat{k}` says.

    Returns
    -------
    PSISReweighting
        Weights, Pareto :math:`\hat{k}`, ESS and verdict.

    Raises
    ------
    ValueError
        If there are fewer than 2 draws, the log ratio contains ``nan`` or
        ``+inf``, or every draw has zero density under the new model.

    Examples
    --------
    Shift the mean of a standard normal posterior from 0 to 0.5:

    .. code-block:: python

        import numpy as np
        from scipy import stats

        from pymc_marketing.mmm.reweighting import psis_reweight

        theta = np.random.default_rng(0).normal(size=4_000)
        log_ratio = stats.norm(0.5, 1).logpdf(theta) - stats.norm(0, 1).logpdf(theta)
        result = psis_reweight(log_ratio)
        result.verdict  # "use it"
        result.weights @ theta  # close to 0.5
    """
    lr = np.asarray(log_ratio, dtype=np.float64).ravel()
    n_draws = lr.size
    if n_draws < 2:
        raise ValueError("Need at least 2 draws.")
    if np.any(np.isnan(lr)) or np.any(np.isposinf(lr)):
        raise ValueError("The log ratio contains NaN or +inf.")
    finite = np.isfinite(lr)
    if not finite.any():
        raise ValueError("The new model has zero density on every draw.")

    if np.ptp(lr[finite]) <= _CONSTANT_ATOL:
        # Nothing changed, or a pure truncation: there is no tail to smooth.
        weights = finite.astype(np.float64)
        weights /= weights.sum()
        pareto_k = float("nan")
        verdict = _VERDICTS[0]
    else:
        # ArviZ smooths the weights exp(-x) of its input x, so the log ratio
        # goes in negated.
        log_w_finite, k = array_stats.psislw(-lr[finite], r_eff=r_eff)
        log_w = np.full(n_draws, -np.inf)
        log_w[finite] = log_w_finite
        if np.any(np.isnan(log_w)) or np.any(np.isposinf(log_w)):
            raise ValueError("PSIS produced NaN or +inf log weights.")
        weights = np.exp(log_w - logsumexp(log_w))
        pareto_k = float(k)
        # A non-finite k-hat means the fit failed: treat it as the worst case.
        verdict = (
            _VERDICTS[2]
            if not np.isfinite(pareto_k)
            else _VERDICTS[np.searchsorted([K_USE_MAX, K_CHECK_MAX], pareto_k, "right")]
        )

    ess = float(r_eff / np.sum(weights**2))
    if ess < min_ess:
        verdict = _VERDICTS[2]
    return PSISReweighting(
        weights=weights,
        pareto_k=pareto_k,
        ess=ess,
        ess_fraction=ess / n_draws,
        verdict=verdict,
    )


def relative_efficiency(
    posterior: xr.Dataset | xr.DataTree, var_names: Sequence[str] | None = None
) -> float:
    """Relative MCMC efficiency of the reference draws, for ``r_eff``.

    The mean ESS (``method="mean"``, as ``az.loo`` uses) of the variables over
    the number of draws, capped at 1 so that anticorrelated draws do not shorten
    the Pareto tail that PSIS fits.

    Parameters
    ----------
    posterior : xarray.Dataset or xarray.DataTree
        Posterior with ``chain`` and ``draw`` dimensions, e.g.
        ``mmm.idata["posterior"]``.
    var_names : sequence of str, optional
        Variables to average over. All variables by default.

    Returns
    -------
    float
        Value in ``(0, 1]``, to pass as ``r_eff`` to :func:`psis_reweight`.
    """
    dataset = (
        posterior.to_dataset() if isinstance(posterior, xr.DataTree) else posterior
    )
    if var_names is not None:
        dataset = dataset[list(var_names)]
    ess = az.ess(dataset, method="mean")
    n_draws = dataset.sizes["chain"] * dataset.sizes["draw"]
    mean_ess = np.mean(np.concatenate([da.values.ravel() for da in ess.values()]))
    return min(float(mean_ess / n_draws), 1.0)
