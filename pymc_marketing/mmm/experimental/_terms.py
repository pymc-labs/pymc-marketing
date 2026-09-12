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
"""Graph terms binding existing MMM components to the labeled dataset."""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import pymc.dims as pmd
import xarray as xr
from pymc_extras.prior import VariableFactory
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.components.adstock import AdstockTransformation, NoAdstock
from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.experimental._data import _align_labels, _dates
from pymc_marketing.mmm.experimental._graph import (
    BuildContext,
    GraphTerm,
    _copy_recipe_value,
    _dimensions,
)
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.transformers import ConvMode


class MediaTransform(GraphTerm):
    """Apply configured transformations along ``date`` to a media expression.

    Parameters
    ----------
    media : object
        Expression producing the raw media tensor, typically ``Data("spend")``.
        Dimensions other than ``date`` are carried through unchanged.
    *transforms : Transformation
        Configured adstock or saturation instances applied left to right.
        Parameter priors keep their own declared dimensions, which must be dataset
        dimensions. ``DataArray`` constants are aligned to dataset labels by name.

    Notes
    -----
    Transformation variable names must be distinct within one model; set ``prefix``
    when the same transformation kind appears twice.
    ``required_history`` totals the causal lookback of every adstock stage, and
    forecasts prepend that many training rows.
    Reduce over ``channel`` explicitly, for example with ``Transform``; the
    equation output never sums dimensions implicitly.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.experimental import Data, MediaTransform

        response = MediaTransform(
            Data("spend"),
            GeometricAdstock(l_max=8),
            LogisticSaturation(priors={"beta": Prior("HalfNormal", dims="channel")}),
        )
    """

    def __init__(self, media: Any, *transforms: Transformation) -> None:
        if not transforms or any(
            not isinstance(transform, Transformation) for transform in transforms
        ):
            raise TypeError(
                "MediaTransform requires one or more Transformation instances."
            )
        names = [
            name
            for transform in transforms
            for name in transform.variable_mapping.values()
        ]
        if len(set(names)) != len(names):
            raise ValueError(
                "Transformation variable names must be distinct; set a distinct prefix."
            )
        self.media = media
        self.transforms = tuple(transforms)

    def dependencies(self) -> tuple[Any, ...]:
        """Return the media expression."""
        return (self.media,)

    def _specification_state(self) -> Any:
        return self.media, self.transforms

    @property
    def required_history(self) -> int:
        """Return the total causal adstock lookback in observation periods."""
        total = 0
        for transform in self.transforms:
            if not isinstance(transform, AdstockTransformation) or isinstance(
                transform, NoAdstock
            ):
                continue
            if transform.mode != ConvMode.After:
                raise ValueError(
                    "Adstock history requires causal mode=ConvMode.After; noncausal boundaries are unsupported."
                )
            total += transform.l_max - 1
        return total

    def _build(self, context: BuildContext) -> XTensorVariable:
        value = context.build(self.media)
        if "date" not in getattr(value, "dims", ()):
            raise ValueError(
                "MediaTransform requires a media expression with a date dimension."
            )
        for transform in self.transforms:
            value = transform.function(
                value, dim="date", **self._parameters(context, transform)
            )
        return value

    def _parameters(
        self, context: BuildContext, transform: Transformation
    ) -> dict[str, Any]:
        clone = copy(transform)
        clone.__dict__ = {
            key: _copy_recipe_value(value) for key, value in vars(transform).items()
        }
        for parameter, prior in clone.function_priors.items():
            if isinstance(prior, VariableFactory):
                context._claim_name(clone.variable_mapping[parameter], self)
                if prior.dims:
                    context._ensure_dims(_dimensions(prior.dims))
            elif isinstance(prior, xr.DataArray):
                clone.function_priors[parameter] = pmd.as_xtensor(
                    _align_labels(prior, context.ds)
                )
            elif np.ndim(prior):
                raise ValueError(
                    "Non-scalar transformation constants require a DataArray with named coordinates."
                )
        return clone._create_distributions()


class Seasonality(GraphTerm):
    """Evaluate a configured Fourier seasonality on the dataset ``date`` coordinate.

    Parameters
    ----------
    fourier : FourierBase
        Configured ``YearlyFourier``, ``MonthlyFourier``, or ``WeeklyFourier``.
        Its prior keeps its declared dimensions, which include ``fourier.prefix``.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import YearlyFourier
        from pymc_marketing.mmm.experimental import Seasonality

        seasonality = Seasonality(
            YearlyFourier(
                n_order=2, prior=Prior("Laplace", mu=0, b=1, dims=("geo", "fourier"))
            )
        )
    """

    def __init__(self, fourier: FourierBase) -> None:
        if not isinstance(fourier, FourierBase):
            raise TypeError("Seasonality requires a configured Fourier component.")
        self.fourier = fourier

    def _specification_state(self) -> Any:
        return self.fourier

    def _build(self, context: BuildContext) -> XTensorVariable:
        if "date" not in context.ds.dims:
            raise ValueError("Seasonality requires a date dimension in the dataset.")
        context._ensure_dims(("date",))
        fourier = self.fourier.model_copy(
            update={"prior": _copy_recipe_value(self.fourier.prior)}
        )
        context._claim_name(fourier.variable_name, self)
        context._ensure_dims(
            [dim for dim in _dimensions(fourier.prior.dims) if dim != fourier.prefix]
        )
        data_name = f"_{fourier.prefix}_dayofperiod"
        context._claim_name(data_name, self)
        dates = _dates(context.ds.coords["date"].values)
        days = pmd.Data(
            data_name, fourier._get_days_in_period(dates).to_numpy(), dims="date"
        )
        return fourier.apply(days)
