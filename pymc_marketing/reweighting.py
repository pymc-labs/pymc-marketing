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
from typing import Literal

import arviz as az
import numpy as np
import numpy.typing as npt
import xarray as xr
from arviz_stats.base import array_stats
from scipy.special import logsumexp

__all__ = ["PSISReweighting", "psis_reweight", "relative_efficiency"]

Verdict = Literal["use it", "check", "refit"]

#: Pareto :math:`\hat{k}` below which the reweighted posterior can be used.
K_USE_MAX = 0.5

#: Largest refit threshold for :math:`\hat{k}`. The threshold is
#: :math:`\min(1 - 1/\log_{10} S, 0.7)` for :math:`S` finite draws, so it is
#: lower below about 2,150 draws.
K_CHECK_MAX = 0.7

#: Default ESS below which the verdict is "refit": about 10 effective draws in
#: each tail of a 94% interval.
MIN_ESS = 400.0

#: Default ESS fraction below which "use it" becomes "check": the weights rest
#: on few of the draws, so intervals can be off even when the mean is right.
MIN_ESS_FRACTION = 0.05

#: Fewest finite draws for which PSIS can fit a Pareto tail.
MIN_PSIS_DRAWS = 25

# Spread of the log ratio below which it counts as constant: absolute, plus
# relative to its magnitude for rounding noise in differences of large sums.
_CONSTANT_ATOL = 1e-10
_CONSTANT_RTOL = 1e-12


@dataclass(frozen=True, eq=False)
class PSISReweighting:
    r"""Smoothed weights of a reweighting and its diagnostics.

    Attributes
    ----------
    weights : numpy.ndarray
        Normalized importance weights, shape ``(n_draws,)``.
    pareto_k : float
        Pareto :math:`\hat{k}` of the fitted tail. ``nan`` when no tail was
        fitted: the log ratio is constant, its upper tail is tied (the weights
        are bounded), or fewer than ``MIN_PSIS_DRAWS`` draws are finite.
    pareto_k_threshold : float
        :math:`\hat{k}` from which the verdict is ``"refit"``,
        :math:`\min(1 - 1/\log_{10} S, 0.7)` for :math:`S` finite draws (Vehtari et al., 2024).
        ``nan`` below ``MIN_PSIS_DRAWS`` finite draws.
    ess : float
        Effective sample size of the weights, ``r_eff / sum(weights**2)``. The
        factor ``r_eff`` accounts for the autocorrelation of the MCMC draws, as
        in the ``loo`` R package and ArviZ's moment matching.
    ess_fraction : float
        ``ess`` as a fraction of the number of draws.
    verdict : {"use it", "check", "refit"}
        ``"use it"`` below :math:`\hat{k}` 0.5, ``"check"`` from 0.5 to
        ``pareto_k_threshold`` and ``"refit"`` from there on. The ESS can only
        make it stricter: ``"refit"`` below ``min_ess``, and ``"check"``
        instead of ``"use it"`` below ``min_ess_fraction`` (see
        :func:`psis_reweight`). Also ``"refit"`` when too few draws are finite
        to fit a tail. Without a fitted tail (``pareto_k`` is ``nan``), the ESS
        alone decides.
    """

    weights: npt.NDArray[np.float64]
    pareto_k: float
    pareto_k_threshold: float
    ess: float
    ess_fraction: float
    verdict: Verdict


def _is_constant(x: npt.NDArray[np.float64]) -> bool:
    """Whether the spread of ``x`` is rounding noise."""
    return bool(np.ptp(x) <= _CONSTANT_ATOL + _CONSTANT_RTOL * np.max(np.abs(x)))


def _tail_length(n_draws: int, r_eff: float) -> int:
    """Return how many of the largest ratios PSIS fits, as ``arviz_stats`` does."""
    if n_draws * r_eff > 225:
        return int(np.floor(3 * (n_draws / r_eff) ** 0.5))
    return n_draws // 5


def psis_reweight(
    log_ratio: npt.ArrayLike,
    r_eff: float = 1.0,
    min_ess: float = MIN_ESS,
    min_ess_fraction: float = MIN_ESS_FRACTION,
) -> PSISReweighting:
    r"""Pareto-smoothed importance weights of a log importance ratio.

    Parameters
    ----------
    log_ratio : array-like, shape (n_draws,)
        ``log p_new - log p_ref`` per posterior draw, with chains and draws
        merged. Draws with ``-inf`` (zero density under the new model) get
        weight 0 and are left out of the smoothing.
    r_eff : float, default 1.0
        Relative efficiency of the reference draws, in ``(0, 1]``, see
        :func:`relative_efficiency`. It sets the length of the Pareto tail and
        scales the reported ESS.
    min_ess : float, default 400.0
        ESS below which the verdict is ``"refit"`` whatever :math:`\hat{k}` says.
    min_ess_fraction : float, default 0.05
        ESS fraction below which ``"use it"`` becomes ``"check"``.

    Returns
    -------
    PSISReweighting
        Weights, Pareto :math:`\hat{k}`, ESS and verdict.

    Raises
    ------
    ValueError
        If there are fewer than 2 draws, the log ratio contains ``nan`` or
        ``+inf``, every draw has zero density under the new model, ``r_eff`` is
        not in ``(0, 1]``, ``min_ess`` is negative or not finite, or
        ``min_ess_fraction`` is not in ``[0, 1]``.

    Notes
    -----
    :math:`\hat{k}` is itself estimated from the draws: for the same change it
    can move by 0.3 or more between reference fits of the same model. Treat 0.5
    and the refit threshold as zones, not sharp lines, and refit when a decision
    depends on a value near them.

    Examples
    --------
    Shift the mean of a standard normal posterior from 0 to 0.5:

    .. code-block:: python

        import numpy as np
        from scipy import stats

        from pymc_marketing.reweighting import psis_reweight

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
    if not (np.isfinite(r_eff) and 0 < r_eff <= 1):
        raise ValueError(f"r_eff must be in (0, 1], got {r_eff}.")
    if not (np.isfinite(min_ess) and min_ess >= 0):
        raise ValueError(f"min_ess must be finite and non-negative, got {min_ess}.")
    if not (np.isfinite(min_ess_fraction) and 0 <= min_ess_fraction <= 1):
        raise ValueError(f"min_ess_fraction must be in [0, 1], got {min_ess_fraction}.")
    finite = np.isfinite(lr)
    if not finite.any():
        raise ValueError("The new model has zero density on every draw.")

    lr_finite = lr[finite]
    n_finite = lr_finite.size
    pareto_k = float("nan")
    k_threshold = (
        min(1 - 1 / np.log10(n_finite), K_CHECK_MAX)
        if n_finite >= MIN_PSIS_DRAWS
        else float("nan")
    )
    verdict: Verdict
    if _is_constant(lr_finite):
        # Nothing changed, or a pure truncation: the weights are uniform.
        log_w_finite = np.zeros(n_finite)
        verdict = "use it"
    elif n_finite < MIN_PSIS_DRAWS:
        # Too few draws to fit a Pareto tail, so nothing backs the weights.
        log_w_finite = lr_finite
        verdict = "refit"
    elif _is_constant(np.sort(lr_finite)[-_tail_length(n_finite, r_eff) :]):
        # Tied upper tail: the weights are bounded and there is no tail to fit.
        log_w_finite = lr_finite
        verdict = "use it"
    else:
        # ArviZ smooths the weights exp(-x) of its input x, so the log ratio
        # goes in negated.
        log_w_finite, k = array_stats.psislw(-lr_finite, r_eff=r_eff)
        if np.any(np.isnan(log_w_finite)) or np.any(np.isposinf(log_w_finite)):
            raise ValueError("PSIS produced NaN or +inf log weights.")
        pareto_k = float(k)
        # A non-finite k-hat means the fit failed: treat it as the worst case.
        if not np.isfinite(pareto_k) or pareto_k >= k_threshold:
            verdict = "refit"
        elif pareto_k >= K_USE_MAX:
            verdict = "check"
        else:
            verdict = "use it"

    log_w = np.full(n_draws, -np.inf)
    log_w[finite] = log_w_finite
    weights = np.exp(log_w - logsumexp(log_w))
    ess = float(r_eff / np.sum(weights**2))
    ess_fraction = ess / n_draws
    if ess < min_ess:
        verdict = "refit"
    elif verdict == "use it" and ess_fraction < min_ess_fraction:
        verdict = "check"
    return PSISReweighting(
        weights=weights,
        pareto_k=pareto_k,
        pareto_k_threshold=float(k_threshold),
        ess=ess,
        ess_fraction=ess_fraction,
        verdict=verdict,
    )


def relative_efficiency(
    posterior: xr.Dataset | xr.DataTree, var_names: str | Sequence[str] | None = None
) -> float:
    """Relative MCMC efficiency of the reference draws, for ``r_eff``.

    The mean ESS (``method="mean"``, as ``az.loo`` uses) over the number of
    draws, capped at 1 so that anticorrelated draws do not shorten the Pareto
    tail that PSIS fits.

    The mean runs over every element of every selected variable, so a
    date-indexed variable such as ``mu`` outweighs a per-channel one, and a
    constant counts as fully efficient. It is an average: the efficiency of a
    single quantity, such as the ROAS of one channel, can be lower, so pass the
    variables the reweighting is about.

    Parameters
    ----------
    posterior : xarray.Dataset or xarray.DataTree
        Posterior with ``chain`` and ``draw`` dimensions, e.g.
        ``mmm.idata["posterior"]``.
    var_names : str or sequence of str, optional
        Variables to average over. All variables by default.

    Returns
    -------
    float
        Value in ``(0, 1]``, to pass as ``r_eff`` to :func:`psis_reweight`.

    Raises
    ------
    ValueError
        If ``var_names`` is empty, or the posterior has no variables or no
        ``chain`` and ``draw`` dimensions.
    """
    dataset = (
        posterior.to_dataset() if isinstance(posterior, xr.DataTree) else posterior
    )
    if isinstance(var_names, str):
        var_names = [var_names]
    if var_names is not None:
        if len(var_names) == 0:
            raise ValueError("var_names is empty.")
        dataset = dataset[list(var_names)]
    if not dataset.data_vars or not {"chain", "draw"} <= set(dataset.dims):
        raise ValueError(
            "Need posterior variables with chain and draw dimensions, "
            "e.g. idata['posterior']."
        )
    ess = az.ess(dataset, method="mean")
    n_draws = dataset.sizes["chain"] * dataset.sizes["draw"]
    mean_ess = np.mean(np.concatenate([da.values.ravel() for da in ess.values()]))
    return min(float(mean_ess / n_draws), 1.0)
