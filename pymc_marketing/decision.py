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
"""Decision summaries of posterior draws along a labelled dimension.

The functions take draws of one quantity, with a dimension whose labels are
the options to compare: the variants of an experiment, the steps of a funnel,
the segments of a customer base. They answer three questions.

* :func:`contrast`: how far is each option from a reference option, draw by draw?
* :func:`prob_best`: with what posterior probability is each option the best one?
* :func:`expected_loss`: how much is expected to be lost by picking an option
  when another one is better?

The draws carry either the dimensions ``("chain", "draw")`` or a single
``"sample"`` dimension. Every other dimension is kept, and the summaries are
computed separately for each of its labels.

Examples
--------
Compare the conversion rate of two variants:

.. code-block:: python

    import numpy as np
    import xarray as xr

    from pymc_marketing.decision import contrast, expected_loss, prob_best

    rng = np.random.default_rng(0)
    rate = xr.DataArray(
        np.stack([rng.beta(21, 181, 4_000), rng.beta(31, 171, 4_000)], axis=-1),
        dims=("sample", "variant"),
        coords={"variant": ["control", "treatment"]},
    )

    lift = contrast(rate, "variant", reference="control")
    prob_best(rate, "variant")
    expected_loss(rate, "variant")
"""

from collections.abc import Hashable
from typing import Literal

import xarray as xr

__all__ = ["contrast", "expected_loss", "prob_best"]


def _sample_dims(samples: xr.DataArray) -> tuple[str, ...]:
    """Return the dimensions of ``samples`` that index the posterior draws."""
    if "chain" in samples.dims and "draw" in samples.dims:
        return ("chain", "draw")
    if "sample" in samples.dims:
        return ("sample",)
    raise ValueError(
        "samples must have the dimensions ('chain', 'draw') or 'sample'; "
        f"got {samples.dims}."
    )


def _validate(samples: xr.DataArray, dim: str) -> tuple[str, ...]:
    """Check that ``samples`` can be compared over ``dim`` and return the sample dimensions."""
    sample_dims = _sample_dims(samples)
    if dim in sample_dims:
        raise ValueError(f"{dim!r} is a sample dimension and cannot be compared over.")
    if dim not in samples.dims:
        raise ValueError(f"{dim!r} is not a dimension of samples; got {samples.dims}.")
    if samples.sizes[dim] < 2:
        raise ValueError(
            f"{dim!r} needs at least two labels to compare; got {samples.sizes[dim]}."
        )
    if samples.isnull().any():
        raise ValueError("samples contain NaN.")
    return sample_dims


def contrast(
    samples: xr.DataArray,
    dim: str,
    reference: Hashable,
    *,
    kind: Literal["difference", "ratio"] = "difference",
) -> xr.DataArray:
    """Compare every label of a dimension with a reference label, draw by draw.

    Parameters
    ----------
    samples : xarray.DataArray
        Posterior draws with the dimensions ``("chain", "draw")`` or ``"sample"``,
        and the dimension ``dim``.
    dim : str
        Dimension whose labels are compared.
    reference : Hashable
        Label of ``dim`` the others are compared with, for example the control
        variant.
    kind : {"difference", "ratio"}, default "difference"
        ``"difference"`` returns ``samples - reference``. ``"ratio"`` returns
        ``samples / reference``; subtract one for the relative lift.

    Returns
    -------
    xarray.DataArray
        Draws with the dimensions of ``samples``. Along ``dim`` the reference
        label is dropped and the other labels keep their order. ``attrs``
        records ``reference`` and ``kind``.

    Raises
    ------
    ValueError
        If ``samples`` has no sample dimensions or contains NaN, if ``dim`` is
        missing, is a sample dimension or has fewer than two labels, if
        ``reference`` is not a label of ``dim``, or if ``kind`` is unknown.

    Examples
    --------
    Posterior probability that the treatment converts better than the control:

    .. code-block:: python

        lift = contrast(rate, "variant", reference="control")
        (lift.sel(variant="treatment") > 0).mean()
    """
    if kind not in ("difference", "ratio"):
        raise ValueError(f"kind must be 'difference' or 'ratio'; got {kind!r}.")
    _validate(samples, dim)
    labels = samples.get_index(dim)
    if reference not in labels:
        raise ValueError(
            f"{reference!r} is not a label of {dim!r}; got {list(labels)}."
        )

    baseline = samples.sel({dim: reference}, drop=True)
    others = samples.drop_sel({dim: reference})
    result = others - baseline if kind == "difference" else others / baseline
    return result.transpose(*samples.dims).assign_attrs(reference=reference, kind=kind)


def prob_best(
    samples: xr.DataArray, dim: str, *, higher_is_better: bool = True
) -> xr.DataArray:
    """Compute the posterior probability that each label of a dimension is the best one.

    A draw in which several labels tie for the best value counts equally for
    each of them, so the probabilities sum to one along ``dim``.

    Parameters
    ----------
    samples : xarray.DataArray
        Posterior draws with the dimensions ``("chain", "draw")`` or ``"sample"``,
        and the dimension ``dim``.
    dim : str
        Dimension whose labels are ranked.
    higher_is_better : bool, default True
        Whether the best label is the one with the largest value (a conversion
        rate) or the smallest (a cost).

    Returns
    -------
    xarray.DataArray
        Probabilities with the dimensions of ``samples`` minus the sample
        dimensions.

    Raises
    ------
    ValueError
        If ``samples`` has no sample dimensions or contains NaN, or if ``dim``
        is missing, is a sample dimension or has fewer than two labels.

    Examples
    --------
    .. code-block:: python

        prob_best(rate, "variant").sel(variant="treatment")
    """
    sample_dims = _validate(samples, dim)
    best = samples.max(dim) if higher_is_better else samples.min(dim)
    is_best = samples == best
    return (is_best / is_best.sum(dim)).mean(sample_dims)


def expected_loss(
    samples: xr.DataArray, dim: str, *, higher_is_better: bool = True
) -> xr.DataArray:
    r"""Compute the expected loss of choosing each label of a dimension.

    The loss of a label in one draw is the gap between the best value in that
    draw and the value of the label, which is zero when the label is the best
    one. Its posterior mean is in the units of ``samples``:

    .. math::

        L_k = \mathbb{E}\left[\max_j \theta_j - \theta_k\right].

    With the default ``higher_is_better=True``, the label with the smallest
    expected loss is the one with the largest posterior mean. The size of the
    loss says how much is at stake in the choice, and a threshold on it is a
    common rule for stopping an experiment.

    Parameters
    ----------
    samples : xarray.DataArray
        Posterior draws with the dimensions ``("chain", "draw")`` or ``"sample"``,
        and the dimension ``dim``.
    dim : str
        Dimension whose labels are the options to choose from.
    higher_is_better : bool, default True
        Whether larger values are better. When ``False`` the loss of a label is
        its value minus the smallest value in the draw.

    Returns
    -------
    xarray.DataArray
        Expected losses with the dimensions of ``samples`` minus the sample
        dimensions.

    Raises
    ------
    ValueError
        If ``samples`` has no sample dimensions or contains NaN, or if ``dim``
        is missing, is a sample dimension or has fewer than two labels.

    Examples
    --------
    .. code-block:: python

        expected_loss(rate, "variant")
    """
    sample_dims = _validate(samples, dim)
    if higher_is_better:
        loss = samples.max(dim) - samples
    else:
        loss = samples - samples.min(dim)
    return loss.transpose(*samples.dims).mean(sample_dims)
