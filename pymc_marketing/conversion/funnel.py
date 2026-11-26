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
"""Conversion funnel model."""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Literal, cast

import numpy.typing as npt
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.conversion import plotting
from pymc_marketing.conversion.data import to_funnel_dataset
from pymc_marketing.decision import prob_best
from pymc_marketing.model_builder import ModelBuilder
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.names import Group
from pymc_marketing.names.conversion import Dim, Var
from pymc_marketing.terms import (
    Named,
    Parameter,
    Transform,
    build_param,
    collect_coords,
    register_data,
)

__all__ = ["FunnelModel", "pooled_p"]


def pooled_p(pooled_dim: str, *, fixed_dims: Sequence[str] = ()) -> Transform:
    r"""Build step probabilities that are partially pooled across one dimension.

    The result is a recipe for the ``p`` entry of the ``model_config`` of
    :class:`FunnelModel`. On the logit scale, each step has a baseline and
    every label of ``pooled_dim`` moves away from it by an effect drawn from a
    common distribution, whose spread is learned for each step:

    .. math::

        \operatorname{logit} p_{k,f,g} &= a_{k,f} + b_{k,g}, \\
        a_{k,f} &\sim \operatorname{Normal}(0, 1.5), \\
        b_{k,g} &\sim \operatorname{Normal}(0, \sigma_k), \\
        \sigma_k &\sim \operatorname{HalfNormal}(1),

    where :math:`k` is the step, :math:`g` a label of ``pooled_dim`` and
    :math:`f` a combination of labels of ``fixed_dims``. A label with few units
    is pulled towards the baseline of its step, and a label with many units
    keeps its own rate.

    The baseline is estimated separately for every label of ``fixed_dims``,
    with no pooling. Put the variants of an experiment there: the difference
    between two variants is then a free contrast for every step.

    Parameters
    ----------
    pooled_dim : str
        Dimension of the data whose labels are partially pooled, for example
        ``"segment"``.
    fixed_dims : sequence of str, optional
        Dimensions of the data that get their own baseline for every step, for
        example ``["variant"]``.

    Returns
    -------
    pymc_marketing.terms.Transform
        The recipe. The posterior holds ``step_logit`` (:math:`a`),
        ``{pooled_dim}_effect`` (:math:`b`), ``{pooled_dim}_effect_sigma``
        (:math:`\sigma`) and ``p``.

    Raises
    ------
    ValueError
        If ``pooled_dim`` is ``"step"`` or is also listed in ``fixed_dims``.

    Examples
    --------
    An A/B test split by customer segment:

    .. code-block:: python

        from pymc_marketing.conversion import FunnelModel, pooled_p

        model = FunnelModel(
            model_config={"p": pooled_p("segment", fixed_dims=["variant"])}
        )
    """
    if pooled_dim == Dim.STEP or pooled_dim in fixed_dims:
        raise ValueError(
            f"pooled_dim must differ from {Dim.STEP!r} and from fixed_dims; "
            f"got {pooled_dim!r} with fixed_dims={list(fixed_dims)}."
        )
    baseline = Parameter(
        "step_logit",
        Prior("Normal", mu=0, sigma=1.5, dims=(Dim.STEP, *fixed_dims)),
    )
    effect = Parameter(
        f"{pooled_dim}_effect",
        Prior(
            "Normal",
            mu=0,
            sigma=Prior("HalfNormal", sigma=1, dims=Dim.STEP),
            dims=(Dim.STEP, pooled_dim),
            centered=False,
        ),
    )
    return Transform(baseline + effect, ptx.math.sigmoid)


class FunnelModel(ModelBuilder):
    r"""Conversion through the steps of a closed funnel.

    A funnel has ordered stages, and a step is the move from one stage into the
    next. For every step :math:`k` and every group :math:`g` (a variant, a
    segment, or a combination of both), the model reads how many units entered
    the step and how many converted:

    .. math::

        \text{converted}_{k,g} \sim \operatorname{Binomial}(\text{entered}_{k,g},\ p_{k,g}).

    In a closed funnel the units that enter a step are the ones that converted
    at the step before it. If each of them moves on independently with
    probability :math:`p_{k,g}`, the product of these Binomials is the
    likelihood of the stage counts, and with one free probability per step
    nothing is lost by modelling the steps one at a time.

    Parameters
    ----------
    model_config : dict, optional
        Dictionary with the key ``"p"``, the conversion probability of a step.
        It is either of the following.

        * A :class:`~pymc_extras.prior.Prior` on the probability scale. Without
          ``dims`` it gets one independent probability for every step and group
          of the data. This is the default, ``Prior("Beta", alpha=1, beta=1)``.
          With ``dims`` it is shared over the dimensions that are left out, for
          example ``dims="step"`` for one probability per step.
        * A recipe built from :mod:`pymc_marketing.terms` whose value is a
          probability. :func:`pooled_p` returns the recipe that partially pools
          the probabilities across one dimension. A recipe on the logit scale
          needs a sigmoid around it. Nothing checks this: values outside
          (0, 1) have zero likelihood, so a recipe without the sigmoid still
          samples, with its prior cut off at the ends of that interval.
    sampler_config : dict, optional
        Keyword arguments for :func:`pymc.sample`.

    Notes
    -----
    The data are the counts of a closed cohort that has had time to convert, in
    the format :func:`~pymc_marketing.conversion.data.to_funnel_dataset`
    describes. Two things make the model too confident or wrong, and neither
    can be seen in the counts.

    * Units that entered shortly before the counts were taken have not had
      time to reach the late stages, which biases the late steps down.
    * Users differ, and counts are often summed over days or campaigns with
      different rates. The Binomial ignores that variation, so the intervals
      are narrower than the data support.

    Every output is a function of the posterior draws of ``p``. A step that no
    unit entered has no data, and its draws are draws from the prior.

    Examples
    --------
    Fit the stage counts of an A/B test and compare the variants:

    .. code-block:: python

        import pandas as pd

        from pymc_marketing.conversion import FunnelModel, to_funnel_dataset
        from pymc_marketing.decision import expected_loss, prob_best

        stages = ["visit", "signup", "paid"]
        df = pd.DataFrame(
            {
                "stage": stages * 2,
                "variant": ["control"] * 3 + ["treatment"] * 3,
                "count": [1000, 300, 60, 1000, 330, 75],
            }
        )

        model = FunnelModel()
        model.fit(to_funnel_dataset(df, stage_order=stages))

        overall = model.overall_conversion()
        prob_best(overall, "variant")
        expected_loss(overall, "variant")
        model.prob_largest_leak()
        model.value_per_entrant(value=40.0)
    """

    _model_type = "FunnelModel"
    version = "0.0.1"

    def __init__(
        self,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ) -> None:
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        # Restore Prior objects and recipes from the dicts that save and load produce.
        self.model_config = parse_model_config(self.model_config)
        self.data: xr.Dataset | None = None

    @property
    def default_model_config(self) -> dict:
        """One independent, uniform conversion probability for every step and group."""
        return {"p": Prior("Beta", alpha=1, beta=1)}

    @property
    def default_sampler_config(self) -> dict:
        """Use the defaults of :func:`pymc.sample`."""
        return {}

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def build_model(self, data: pd.DataFrame | xr.Dataset | None = None) -> None:  # type: ignore[override]
        """Build the PyMC model, available afterwards as ``self.model``.

        Parameters
        ----------
        data : pandas.DataFrame or xarray.Dataset, optional
            Funnel counts in a format that
            :func:`~pymc_marketing.conversion.data.to_funnel_dataset` accepts.
            A ``DataFrame`` of stage counts needs an ordered categorical
            ``stage`` column; otherwise convert it first and pass the
            ``Dataset``. When omitted, the data stored with a fitted model are
            used.

        Raises
        ------
        ValueError
            If there is no data, or if ``p`` has a dimension the data lack.
        """
        if data is not None:
            ds = to_funnel_dataset(data)
        elif self.idata is not None and Group.FIT_DATA in self.idata.children:
            ds = to_funnel_dataset(self.idata[f"/{Group.FIT_DATA}"].to_dataset())
        else:
            raise ValueError(
                "No data to build the model from. Pass data to fit() or build_model()."
            )
        self.data = ds

        dims = ds[Var.CONVERTED].dims
        self.model = pm.Model(coords={dim: ds[dim].values for dim in dims})
        with self.model:
            entered = pmd.Data(Var.ENTERED, ds[Var.ENTERED].values, dims=dims)
            converted = pmd.Data(Var.CONVERTED, ds[Var.CONVERTED].values, dims=dims)
            p = self._build_p(ds)
            pmd.Binomial(Var.Y, n=entered, p=p, observed=converted)

    def _build_p(self, ds: xr.Dataset) -> XTensorVariable:
        """Create the variable ``p`` from ``model_config`` inside the model context."""
        dims = ds[Var.CONVERTED].dims
        spec = self.model_config["p"]
        if isinstance(spec, Prior):
            if not spec.dims:
                # Work on a copy so that the caller's Prior keeps no trace of this dataset.
                spec = spec.deepcopy()
                spec.dims = dims
            self._check_dims(spec.dims, dims)
            return spec.create_variable("p", xdist=True)

        name = getattr(spec, "name", None)
        if name is None:
            spec = Named("p", spec)
        elif name != "p":
            raise ValueError(
                f"The recipe for 'p' is named {name!r}. Name it 'p' or leave it "
                "unnamed, so that the posterior holds a variable 'p'."
            )
        for coord, values in collect_coords(spec, ds=ds).items():
            if coord not in self.model.coords:
                self.model.add_coord(coord, values)
        register_data(spec, ds=ds)
        try:
            p = cast("XTensorVariable", build_param(spec, "p"))
        except ValueError as error:
            error.add_note(
                f"FunnelModel was building 'p' for data with the dimensions {tuple(dims)}. "
                "'p' can only vary over these dimensions."
            )
            raise
        self._check_dims(p.dims, dims)
        return p

    @staticmethod
    def _check_dims(p_dims: Iterable[Hashable], data_dims: Iterable[Hashable]) -> None:
        """Refuse a ``p`` that would broadcast the likelihood beyond the data."""
        data_dims = tuple(data_dims)
        unknown = [dim for dim in p_dims if dim not in data_dims]
        if unknown:
            raise ValueError(
                f"'p' has the dimensions {unknown}, which the data lack. "
                f"The data have {data_dims}."
            )

    def _prepare_fit(self, data: pd.DataFrame | xr.Dataset | None = None) -> None:
        """Build the model for this fit; a new dataset replaces the previous one."""
        self.build_model(data if data is not None else self.data)

    def build_from_idata(self, idata: xr.DataTree) -> None:
        """Rebuild the model from the data stored in ``idata``.

        Used by :meth:`load`.
        """
        self.idata = idata
        self.build_model()

    def step_conversion(self) -> xr.DataArray:
        """Posterior draws of the conversion probability of every step.

        Returns
        -------
        xarray.DataArray
            Draws with the dimensions ``(chain, draw, step, *groups)``, where
            the groups are the other dimensions of ``p``.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        p = self.fit_result["p"]
        if Dim.STEP not in p.dims:
            steps = self.idata[f"/{Group.FIT_DATA}"][Dim.STEP]  # type: ignore[index]
            p = p.expand_dims({Dim.STEP: steps.values})
        return p.transpose("chain", "draw", Dim.STEP, ...)

    def cumulative_conversion(self) -> xr.DataArray:
        """Posterior draws of the share of entrants that reaches each stage.

        The value at a step is the product of the step conversions up to and
        including it: the share of the units that entered the funnel and got
        as far as the stage the step leads into.

        Returns
        -------
        xarray.DataArray
            Draws with the dimensions ``(chain, draw, step, *groups)``.
        """
        return self.step_conversion().cumprod(Dim.STEP)

    def overall_conversion(self) -> xr.DataArray:
        """Posterior draws of the share of entrants that gets through every step.

        Returns
        -------
        xarray.DataArray
            Draws with the dimensions ``(chain, draw, *groups)``.
        """
        return self.step_conversion().prod(Dim.STEP)

    def share_lost(self) -> xr.DataArray:
        """Posterior draws of the share of all entrants that is lost at each step.

        A unit is lost at a step when it reaches the step and does not convert.
        The share is taken over the units that entered the funnel, so it weights
        the drop-off rate of a step, ``1 - p``, by how many units get that far.
        The shares of all steps and the overall conversion add up to one.

        Returns
        -------
        xarray.DataArray
            Draws with the dimensions ``(chain, draw, step, *groups)``.
        """
        step = self.step_conversion()
        reached = step.cumprod(Dim.STEP).shift({Dim.STEP: 1}, fill_value=1.0)
        return reached * (1 - step)

    def prob_largest_leak(self) -> xr.DataArray:
        """Posterior probability that each step is where the funnel loses most entrants.

        The steps are compared by :meth:`share_lost`, the share of all entrants
        lost at the step. A step with a low conversion rate that few units reach
        can therefore rank below an early step with a better rate.

        A funnel with a single step has one place to lose entrants, and the
        probability of that step is one.

        Returns
        -------
        xarray.DataArray
            Probabilities with the dimensions ``(step, *groups)``. They sum to
            one over ``step``.
        """
        share_lost = self.share_lost()
        if share_lost.sizes[Dim.STEP] == 1:
            return xr.ones_like(share_lost.mean(("chain", "draw")))
        return prob_best(share_lost, Dim.STEP)

    def value_per_entrant(self, value: float | xr.DataArray) -> xr.DataArray:
        """Posterior draws of the expected value of one unit that enters the funnel.

        Parameters
        ----------
        value : float or xarray.DataArray
            Value of one unit that gets through every step, for example the
            revenue of a purchase. A ``DataArray`` can differ by group, or
            carry draws with the dimensions ``(chain, draw)`` to propagate the
            uncertainty of the value. Its coordinates have to match the ones of
            :meth:`overall_conversion`.

        Returns
        -------
        xarray.DataArray
            ``value`` times :meth:`overall_conversion`.

        Raises
        ------
        ValueError
            If ``value`` has labels along a shared dimension that differ from
            the ones of :meth:`overall_conversion`.
        """
        overall = self.overall_conversion()
        if isinstance(value, xr.DataArray):
            overall, value = xr.align(overall, value, join="exact")
        return overall * value

    def plot_funnel(
        self,
        *,
        quantity: Literal["cumulative", "step"] = "cumulative",
        **kwargs: Any,
    ) -> tuple[Figure, npt.NDArray[Axes]]:
        """Plot the conversion by step with highest density intervals.

        Parameters
        ----------
        quantity : {"cumulative", "step"}, default "cumulative"
            ``"cumulative"`` plots :meth:`cumulative_conversion`, the share of
            entrants that reaches each stage, which gives the funnel its shape.
            ``"step"`` plots :meth:`step_conversion`.
        **kwargs
            Passed to :func:`~pymc_marketing.conversion.plotting.plot_funnel`:
            ``hue``, ``hdi_prob`` and ``figsize``.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.
        numpy.ndarray of matplotlib.axes.Axes
            One Axes per panel.

        Raises
        ------
        ValueError
            If ``quantity`` is neither ``"cumulative"`` nor ``"step"``.
        """
        if quantity == "cumulative":
            samples = self.cumulative_conversion()
            label = "cumulative conversion"
        elif quantity == "step":
            samples = self.step_conversion()
            label = "step conversion"
        else:
            raise ValueError(
                f"quantity must be 'cumulative' or 'step'; got {quantity!r}."
            )
        fig, axes = plotting.plot_funnel(samples, **kwargs)
        for ax in axes:
            ax.set_xlabel(label)
        return fig, axes
