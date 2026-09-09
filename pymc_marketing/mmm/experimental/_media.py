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
"""Lazy channel handles using the existing adstock and saturation functions."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import copy
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pymc_extras.prior import VariableFactory
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.components.adstock import AdstockTransformation, NoAdstock
from pymc_marketing.mmm.components.base import Transformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation
from pymc_marketing.mmm.experimental._data import _align_labels
from pymc_marketing.mmm.experimental._graph import (
    BuildContext,
    Data,
    Equation,
    GraphTerm,
    _copy_recipe_value,
)
from pymc_marketing.mmm.transformers import ConvMode


def _copy_transform(transform: Transformation, dims: tuple[str, ...]) -> Transformation:
    """Clone a configured transformation, assigning only unspecified prior dimensions."""
    clone = copy(transform)
    clone.__dict__ = {
        key: _copy_recipe_value(value) for key, value in vars(transform).items()
    }
    for prior in clone.function_priors.values():
        if isinstance(prior, VariableFactory) and prior.dims is None:
            try:
                prior.dims = (*dims, "channel")
            except AttributeError as error:
                raise ValueError(
                    "Configure dimensions on the underlying prior when a transformation factory has read-only dims."
                ) from error
    unsupported = set(clone.combined_dims) - {*dims, "channel"}
    if unsupported:
        raise ValueError(
            f"Transformation parameter dimensions are not media dimensions: {sorted(unsupported)!r}."
        )
    return clone


class _TransformParameters(GraphTerm):
    """One identity-cached set of transformation parameters shared by all channels."""

    def __init__(
        self,
        transform: Transformation,
        channels: tuple[str, ...],
        dims: tuple[str, ...],
    ) -> None:
        self.transform = transform
        self.channels = channels
        self.dims = dims

    def dependencies(self) -> tuple[Any, ...]:
        return ()

    def _specification_state(self) -> Any:
        return self.transform, self.channels, self.dims

    def _build(self, context: BuildContext) -> dict[str, Any]:
        model = pm.modelcontext(None)
        if "channel" not in model.coords:
            model.add_coord("channel", self.channels)
        elif tuple(model.coords["channel"]) != self.channels:
            raise ValueError(
                "Model channel coordinates must match the declared media channel order."
            )
        transform = _copy_transform(self.transform, self.dims)
        reference = context.ds.assign_coords(channel=list(self.channels))
        for parameter, prior in transform.function_priors.items():
            name = transform.variable_mapping[parameter]
            if hasattr(prior, "create_variable") and name in model.named_vars:
                raise ValueError(
                    f"Transformation parameter name {name!r} is already owned by another term."
                )
            if isinstance(prior, xr.DataArray):
                transform.function_priors[parameter] = pmd.as_xtensor(
                    _align_labels(prior, reference)
                )
            elif isinstance(prior, (list, tuple, np.ndarray)):
                array = np.asarray(prior)
                if array.ndim:
                    raise ValueError(
                        "Non-scalar transformation constants require a DataArray with named coordinates."
                    )
        return transform._create_distributions()


class _ChannelValue(GraphTerm):
    """A stable raw-valued reference to a channel's current equation or data."""

    def __init__(self, channel: "_Channel") -> None:
        self.channel = channel
        self.data = Data(channel.name)

    def dependencies(self) -> tuple[Any, ...]:
        return (
            self.channel.equation if self.channel.equation is not None else self.data,
        )

    def _specification_state(self) -> Any:
        return self.channel.name, self.dependencies()

    def _build(self, context: BuildContext) -> XTensorVariable:
        return context.build(self.dependencies()[0])


class _ChannelContribution(GraphTerm):
    """Apply the shared pipeline to this channel only, avoiding unrelated dependencies."""

    def __init__(self, channel: "_Channel", media: "Media") -> None:
        self.channel = channel
        self.media = media

    def dependencies(self) -> tuple[Any, ...]:
        return (self.channel.value, *self.media._parameters)

    def _specification_state(self) -> Any:
        return self.channel.name, self.dependencies(), self.media.channel_scale

    def _build(self, context: BuildContext) -> XTensorVariable:
        value = context.build(self.channel.value)
        if set(value.dims) != {"date", *self.media.dims}:
            raise ValueError(
                f"Channel {self.channel.name!r} must have dimensions {('date', *self.media.dims)!r}."
            )
        value = value.transpose("date", *self.media.dims)
        scale = self.media._scale(context, channel=self.channel.name)
        value = value / scale
        channel_index = self.media._channels.index(self.channel.name)
        for transform, bundle in zip(
            self.media.transforms, self.media._parameters, strict=True
        ):
            parameters = context.build(bundle)
            selected = {
                name: parameter[{"channel": channel_index}]
                if "channel" in getattr(parameter, "dims", ())
                else parameter
                for name, parameter in parameters.items()
            }
            value = transform.function(value, dim="date", **selected)
        return value.transpose("date", *self.media.dims)


class _MediaTotal(GraphTerm):
    """Vectorize the default media pipeline and sum its channel dimension."""

    def __init__(self, media: "Media") -> None:
        self.media = media

    def dependencies(self) -> tuple[Any, ...]:
        return (
            *[channel.value for channel in self.media.values()],
            *self.media._parameters,
        )

    def _specification_state(self) -> Any:
        return (
            self.media._channels,
            self.media.dims,
            self.dependencies(),
            self.media.channel_scale,
        )

    def _build(self, context: BuildContext) -> XTensorVariable:
        if not self.media:
            return pmd.as_xtensor(0.0)
        values = []
        for channel in self.media.values():
            value = context.build(channel.value)
            if set(value.dims) != {"date", *self.media.dims}:
                raise ValueError(
                    f"Channel {channel.name!r} must have dimensions {('date', *self.media.dims)!r}."
                )
            values.append(value.transpose("date", *self.media.dims))
        value = pmd.concat(values, dim="channel").transpose(
            "date", *self.media.dims, "channel"
        )
        value = value / self.media._scale(context)
        for transform, bundle in zip(
            self.media.transforms, self.media._parameters, strict=True
        ):
            value = transform.function(value, dim="date", **context.build(bundle))
        return value.transpose("date", *self.media.dims, "channel").sum(dim="channel")


class _Channel:
    """Expose stable raw-value and contribution terms for one named channel."""

    def __init__(self, name: str, media: "Media") -> None:
        self.name = name
        self._media = media
        self._equation: Equation | None = None
        self.value = _ChannelValue(self)
        self.contribution = _ChannelContribution(self, media)

    @property
    def equation(self) -> Equation | None:
        """The channel's stochastic mechanism, or ``None`` for raw input data."""
        return self._equation

    @equation.setter
    def equation(self, equation: Equation | None) -> None:
        if equation is not None and not isinstance(equation, Equation):
            raise TypeError("A media equation must be an Equation or None.")
        if equation is self._equation:
            return
        self._equation = equation
        if self._media._on_change is not None:
            self._media._on_change()


class Media(Mapping[str, _Channel]):
    """A mapping of channel names to lazy, stable value and contribution terms.

    Parameters
    ----------
    channels : sequence of str
        Ordered channel names.
    transforms : tuple of Transformation
        Exactly one configured adstock and one configured saturation, in application order.
        Configurations are copied before assigning default parameter dimensions.
    dims : tuple of str, optional
        Panel dimensions in addition to date and channel.
    on_change : callable, optional
        Internal invalidation callback when a channel equation is replaced.

    Notes
    -----
    Channel values always have raw units, including stochastic endogenous channels.
    Scaling is applied exactly once, downstream of their value terms.
    Saturations requiring unscaled input bypass the fitted channel divisor.
    """

    def __init__(
        self,
        channels: Sequence[str],
        transforms: tuple[Transformation, ...],
        *,
        dims: tuple[str, ...] = (),
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._channels = tuple(channels)
        if len(set(self._channels)) != len(self._channels) or not all(
            isinstance(channel, str) and channel for channel in self._channels
        ):
            raise ValueError("Media channel names must be unique, nonempty strings.")
        if (
            not isinstance(transforms, tuple)
            or len(transforms) != 2
            or sum(
                isinstance(transform, AdstockTransformation) for transform in transforms
            )
            != 1
            or sum(
                isinstance(transform, SaturationTransformation)
                for transform in transforms
            )
            != 1
        ):
            raise TypeError(
                "media_transform must be a tuple containing exactly one adstock and one saturation."
            )
        self.dims = tuple(dims)
        if len(set(self.dims)) != len(self.dims) or {"date", "channel"} & set(
            self.dims
        ):
            raise ValueError(
                "Media panel dimensions must be unique and exclude date and channel."
            )
        self._transforms = tuple(
            _copy_transform(transform, self.dims) for transform in transforms
        )
        self._on_change = on_change
        self.channel_scale: xr.DataArray | float = 1.0
        self._parameters = tuple(
            _TransformParameters(transform, self._channels, self.dims)
            for transform in self.transforms
        )
        self._handles = {name: _Channel(name, self) for name in self._channels}
        self.total = _MediaTotal(self)

    def __getitem__(self, name: str) -> _Channel:
        return self._handles[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._handles)

    def __len__(self) -> int:
        return len(self._handles)

    @property
    def transforms(self) -> tuple[Transformation, ...]:
        """Return the configured transformations in application order."""
        return self._transforms

    @property
    def equations(self) -> dict[str, Equation]:
        """Return attached equations keyed by their channel binding."""
        return {
            name: channel.equation
            for name, channel in self.items()
            if channel.equation is not None
        }

    @property
    def required_history(self) -> int:
        """Return the finite causal adstock history, rejecting unsupported boundaries."""
        adstock = next(
            transform
            for transform in self.transforms
            if isinstance(transform, AdstockTransformation)
        )
        if isinstance(adstock, NoAdstock):
            return 0
        if adstock.mode != ConvMode.After:
            raise ValueError(
                "Adstock history requires causal mode=ConvMode.After; noncausal boundaries are unsupported."
            )
        return adstock.l_max - 1

    def _scale(
        self, context: BuildContext, *, channel: str | None = None
    ) -> XTensorVariable:
        """Align the fitted divisor by labels before converting to a named tensor."""
        if any(
            getattr(transform, "requires_unscaled_input", False)
            for transform in self.transforms
        ):
            return pmd.as_xtensor(1.0)
        scale = self.channel_scale
        if isinstance(scale, xr.DataArray):
            reference = context.ds.assign_coords(channel=list(self._channels))
            if not set(scale.dims).issubset({*self.dims, "channel"}):
                raise ValueError(
                    "Channel scale may only use media panel and channel dimensions."
                )
            scale = _align_labels(scale, reference)
            if channel is not None and "channel" in scale.dims:
                scale = scale.sel(channel=channel, drop=True)
            values = scale.values
        else:
            values = scale
        if not np.isfinite(values).all() or np.any(np.asarray(values) == 0):
            raise ValueError("Channel scaling divisors must be finite and nonzero.")
        return pmd.as_xtensor(scale)
