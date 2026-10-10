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
"""Graph terms binding media transformations and seasonality to the labeled dataset."""

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
from pymc_marketing.mmm.experimental._data import _dates
from pymc_marketing.mmm.experimental._graph import (
    BuildContext,
    GraphTerm,
    _dimensions,
)
from pymc_marketing.mmm.fourier import FourierBase
from pymc_marketing.mmm.transformers import ConvMode


class MediaTransform(GraphTerm):
    """Apply one configured adstock or saturation along ``date``.

    Usually created with ``>>``, which chains stages left to right:
    ``Data("spend") >> GeometricAdstock(l_max=8) >> LogisticSaturation()``.
    Construct it directly when the input is not a graph term, for example
    ``MediaTransform(Data("spend") * 0.001, GeometricAdstock(l_max=8))``.

    Parameters
    ----------
    media : object
        Expression producing the media tensor. It must carry ``date``; its other
        dimensions are carried through in their input order.
    transformation : Transformation
        Configured adstock or saturation instance.

    Notes
    -----
    A parameter prior without ``dims`` takes the input's dimensions other than
    ``date``, matching the stable MMM's default per-channel parameters. Declared
    ``dims`` are kept, and ``dims=()`` shares one value across the input. The
    configured transformation is never modified. ``DataArray`` constants are
    aligned to dataset labels by name.

    Transformation variable names must be distinct within one model; set
    ``prefix`` when the same transformation kind appears twice. Each causal
    adstock stage requires ``l_max - 1`` rows of training history, and forecasts
    prepend the largest total along any dependency path.
    Reduce over ``channel`` explicitly, for example with ``.sum("channel")``; the
    equation output never sums dimensions implicitly.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
        from pymc_marketing.mmm.experimental import Data

        contribution = (
            Data("spend")
            >> GeometricAdstock(l_max=8)
            >> LogisticSaturation(priors={"beta": Prior("HalfNormal", sigma=2)})
        ).named("channel_contribution")
        total = contribution.sum("channel")
    """

    def __init__(self, media: Any, transformation: Transformation) -> None:
        if not isinstance(transformation, Transformation):
            raise TypeError("MediaTransform requires a Transformation instance.")
        self.media = media
        self.transformation = transformation

    def dependencies(self) -> tuple[Any, ...]:
        """Return the media expression."""
        return (self.media,)

    def _specification_state(self) -> Any:
        return self.media, self.transformation

    @property
    def required_history(self) -> int:
        """Return this stage's causal adstock lookback in observation periods."""
        transformation = self.transformation
        if not isinstance(transformation, AdstockTransformation) or isinstance(
            transformation, NoAdstock
        ):
            return 0
        if transformation.mode != ConvMode.After:
            raise ValueError(
                "Adstock history requires causal mode=ConvMode.After; noncausal boundaries are unsupported."
            )
        return transformation.l_max - 1

    def _build(self, context: BuildContext) -> XTensorVariable:
        value = context.build(self.media)
        dims = tuple(getattr(value, "dims", ()))
        if "date" not in dims:
            raise ValueError(
                "MediaTransform requires a media expression with a date dimension."
            )
        parameters = self._parameters(
            context, tuple(dim for dim in dims if dim != "date")
        )
        result = self.transformation.function(value, dim="date", **parameters)
        return result.transpose(*dims, ...)

    def _parameters(
        self, context: BuildContext, default_dims: tuple[str, ...]
    ) -> dict[str, Any]:
        clone = copy(self.transformation)
        clone.__dict__ = {
            key: context._recipe(value)
            for key, value in vars(self.transformation).items()
        }
        for parameter, prior in clone.function_priors.items():
            if isinstance(prior, VariableFactory):
                # Same rule as Transformation.with_default_prior_dims, applied to a
                # copy_prior clone because deepcopy drops Prior core dimensions.
                if prior.dims is None:
                    prior.dims = default_dims
                name = clone.variable_mapping[parameter]
                try:
                    context._claim_name(name, self)
                except ValueError as error:
                    raise ValueError(
                        f"{error} Give repeated transformations distinct prefixes."
                    ) from error
                context._ensure_dims(_dimensions(prior.dims))
            elif isinstance(prior, xr.DataArray):
                clone.function_priors[parameter] = context._constant(prior)
            elif np.ndim(prior):
                raise ValueError(
                    "Non-scalar transformation constants require a DataArray with named coordinates."
                )
        return clone._create_distributions()


class FourierTerm(GraphTerm):
    """Evaluate a Fourier seasonality component on the dataset ``date`` coordinate.

    Not constructed directly: a ``YearlyFourier``, ``MonthlyFourier``, or
    ``WeeklyFourier`` used anywhere in an expression is built through one
    ``FourierTerm`` per component object, so a component shared by several
    equations is one set of parameters. Its prior keeps its declared dimensions,
    which include ``fourier.prefix``.

    Parameters
    ----------
    fourier : FourierBase
        Configured Fourier seasonality component.
    """

    def __init__(self, fourier: FourierBase) -> None:
        self.fourier = fourier

    def _specification_state(self) -> Any:
        return self.fourier

    def _build(self, context: BuildContext) -> XTensorVariable:
        if "date" not in context.ds.dims:
            raise ValueError(
                "Fourier seasonality requires a date dimension in the dataset."
            )
        context._ensure_dims(("date",))
        fourier = self.fourier.model_copy(
            update={"prior": context._recipe(self.fourier.prior)}
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
