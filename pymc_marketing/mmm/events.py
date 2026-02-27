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
"""Event transformations.

This module provides event transformations for use in Marketing Mix Models.

.. plot::
    :context: close-figs

    import numpy as np
    import pandas as pd
    import pymc as pm

    import matplotlib.pyplot as plt

    from pymc_marketing.mmm.events import EventEffect, GaussianBasis
    from pymc_marketing.plot import plot_curve
    from pymc_extras.prior import Prior

    seed = sum(map(ord, "Events"))
    rng = np.random.default_rng(seed)

    df_events = pd.DataFrame(
        {
            "event": ["single day", "multi day"],
            "start_date": pd.to_datetime(["2025-01-01", "2025-01-20"]),
            "end_date": pd.to_datetime(["2025-01-02", "2025-01-25"]),
        }
    )

    def difference_in_days(model_dates, event_dates):
        if hasattr(model_dates, "to_numpy"):
            model_dates = model_dates.to_numpy()
        if hasattr(event_dates, "to_numpy"):
            event_dates = event_dates.to_numpy()

        one_day = np.timedelta64(1, "D")
        return (model_dates[:, None] - event_dates) / one_day


    def create_basis_matrix(df_events: pd.DataFrame, model_dates: np.ndarray):
        start_dates = df_events["start_date"]
        end_dates = df_events["end_date"]

        start_ref = difference_in_days(model_dates, start_dates)
        end_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (start_ref >= 0) & (end_ref <= 0),
            0,
            np.where(np.abs(start_ref) < np.abs(end_ref), start_ref, end_ref),
        )


    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1, dims="event"),
        }
    )
    effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
    effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("event",))

    dates = pd.date_range("2024-12-01", periods=3 * 31, freq="D")

    X = create_basis_matrix(df_events, model_dates=dates)

    coords = {"date": dates, "event": df_events["event"].to_numpy()}
    with pm.Model(coords=coords) as model:
        pm.Deterministic("effect", effect.apply(X), dims=("date", "event"))

        idata = pm.sample_prior_predictive(random_seed=rng)

    fig, axes = idata.prior.effect.pipe(
        plot_curve,
        "date",
        random_seed=rng,
        subplot_kwargs={"ncols": 1},
    )
    fig.suptitle("Gaussian Event Effect")
    plt.show()

"""

from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    model_validator,
    validate_call,
)
from pymc import logp
from pymc_extras.deserialize import deserialize, register_deserialization
from pymc_extras.prior import Prior
from pytensor.xtensor.type import XTensorVariable, as_xtensor

from pymc_marketing.mmm.components.base import Transformation, create_registration_meta
from pymc_marketing.mmm.dims import XTensorLike

BASIS_TRANSFORMATIONS: dict = {}
BasisMeta = create_registration_meta(BASIS_TRANSFORMATIONS)


class Basis(Transformation, metaclass=BasisMeta):  # type: ignore[metaclass]
    """Basis transformation associated with an event model."""

    prefix: str = "basis"
    lookup_name: str

    @validate_call
    def sample_curve(
        self,
        parameters: InstanceOf[xr.Dataset] = Field(
            ..., description="Parameters of the saturation transformation."
        ),
        days: int = Field(
            14,
            gt=0,
            description="Number of days before and after the basis.",
        ),
    ) -> xr.DataArray:
        """Sample the curve of the saturation transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with the parameters of the saturation transformation.
        days : int
            Number of days around basis. Default is 14 days or two weeks before and
            after the basis for a total of 28 days.

        Returns
        -------
        xr.DataArray
            Curve of the saturation transformation.

        """
        x = np.linspace(-days, days, 100)

        coords = {"x": x}

        return self._sample_curve(
            var_name="saturation",
            parameters=parameters,
            x=x,
            coords=coords,
        )


def basis_from_dict(data: dict) -> Basis:
    """Create a basis transformation from a dictionary."""
    data = data.copy()
    lookup_name = data.pop("lookup_name")
    cls = BASIS_TRANSFORMATIONS[lookup_name]

    if "priors" in data:
        data["priors"] = {k: deserialize(v) for k, v in data["priors"].items()}

    return cls(**data)


def _is_basis(data):
    return "lookup_name" in data and data["lookup_name"] in BASIS_TRANSFORMATIONS


register_deserialization(
    is_type=_is_basis,
    deserialize=basis_from_dict,
)


class EventEffect(BaseModel):
    """Event effect associated with an event model."""

    basis: InstanceOf[Basis]
    effect_size: InstanceOf[Prior]
    dims: str | tuple[str, ...]
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    def _dims_to_tuple(self):
        if isinstance(self["dims"], str):
            self["dims"] = (self["dims"],)

        return self

    @model_validator(mode="after")
    def _validate_dims(self):
        if not self.dims:
            raise ValueError("The dims must not be empty.")

        if not set(self.basis.combined_dims).issubset(set(self.dims)):
            raise ValueError("The dims must contain all dimensions of the basis.")

        if not set(self.effect_size.dims or {}).issubset(set(self.dims)):
            raise ValueError("The dims must contain all dimensions of the effect size.")

        return self

    def apply(self, X: XTensorLike, name: str = "event") -> XTensorVariable:
        """Apply the event effect to the data."""
        return self.basis.apply(X) * self.effect_size.create_variable(
            f"{name}_effect_size", xdist=True
        )

    def to_dict(self) -> dict:
        """Convert the event effect to a dictionary."""
        return {
            "class": "EventEffect",
            "data": {
                "basis": self.basis.to_dict(),
                "effect_size": self.effect_size.to_dict(),
                "dims": self.dims,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventEffect":
        """Create an event effect from a dictionary."""
        return cls(
            basis=deserialize(data["basis"]),
            effect_size=deserialize(data["effect_size"]),
            dims=data["dims"],
        )


def _is_event_effect(data: dict) -> bool:
    """Check if the data is an event effect."""
    return data["class"] == "EventEffect"


register_deserialization(
    is_type=_is_event_effect,
    deserialize=lambda data: EventEffect.from_dict(data["data"]),
)


class GaussianBasis(Basis):
    """Gaussian basis transformation."""

    lookup_name = "gaussian"

    def function(
        self, x: XTensorLike, sigma: XTensorLike, *, dim: str | None = None
    ) -> XTensorVariable:
        """Gaussian bump function."""
        x = as_xtensor(x)
        rv = pmd.Normal.dist(mu=0.0, sigma=sigma)
        out = pmd.math.exp(logp(rv, x))
        return out

    default_priors = {
        "sigma": Prior("Gamma", mu=7, sigma=1),
    }


class HalfGaussianBasis(Basis):
    R"""One-sided Gaussian basis transformation.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.events import HalfGaussianBasis
        from pymc_extras.prior import Prior
        half_gaussian = HalfGaussianBasis(
            priors={
                "sigma": Prior("Gamma", mu=[3, 4], sigma=1, dims="event"),
            }
        )
        coords = {"event": ["PyData-Berlin", "PyCon-Finland"]}
        prior = half_gaussian.sample_prior(coords=coords)
        curve = half_gaussian.sample_curve(prior)
        fig, axes = half_gaussian.plot_curve(
            curve, subplot_kwargs={"figsize": (6, 3), "sharey": True}
        )
        for ax in axes:
            ax.set_xlabel("")
        plt.show()

    Parameters
    ----------
    mode : Literal["after", "before"]
        Whether the basis is located before or after the event.
    include_event : bool
        Whether to include the event days in the basis.
    priors : dict[str, Prior]
        Prior for the sigma parameter.
    prefix : str
        Prefix for the parameter names.
    """

    lookup_name = "half_gaussian"

    def __init__(
        self,
        mode: Literal["after", "before"] = "after",
        include_event: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.include_event = include_event

    def function(
        self, x: XTensorLike, sigma: XTensorLike, *, dim: str | None = None
    ) -> XTensorVariable:
        """One-sided Gaussian bump function."""
        x = as_xtensor(x)
        rv = pmd.Normal.dist(mu=0.0, sigma=sigma)
        out = ptx.math.exp(logp(rv, x))
        # Sign determines if the zeroing happens after or before the event.
        sign = 1 if self.mode == "after" else -1
        # Build boolean mask(s) in x's shape and broadcast to out's shape.
        pre_mask = sign * x < 0
        if not self.include_event:
            pre_mask = pre_mask | ptx.math.equal(sign * x, 0)

        return ptx.math.switch(pre_mask, 0, out)

    def to_dict(self) -> dict:
        """Convert the half Gaussian basis to a dictionary."""
        return {
            **super().to_dict(),
            "mode": self.mode,
            "include_event": self.include_event,
        }

    default_priors = {
        "sigma": Prior("Gamma", mu=7, sigma=1),
    }


class AsymmetricGaussianBasis(Basis):
    R"""Asymmetric Gaussian bump basis transformation.

    Allows different widths (sigma_before, sigma_after) and amplitudes (a_after)
    after the event.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        from pymc_marketing.mmm.events import AsymmetricGaussianBasis
        from pymc_extras.prior import Prior
        asy_gaussian = AsymmetricGaussianBasis(
            priors={
                "sigma_before": Prior("Gamma", mu=[3, 4], sigma=1, dims="event"),
                "a_after": Prior("Normal", mu=[-.75, .5], sigma=.2, dims="event"),
            }
        )
        coords = {"event": ["PyData-Berlin", "PyCon-Finland"]}
        prior = asy_gaussian.sample_prior(coords=coords)
        curve = asy_gaussian.sample_curve(prior)
        fig, axes = asy_gaussian.plot_curve(
            curve, subplot_kwargs={"figsize": (6, 3), "sharey": True}
        )
        for ax in axes:
            ax.set_xlabel("")
        plt.show()

    Parameters
    ----------
    event_in : Literal["before", "after", "exclude"]
        Whether to include the event in the before or after part of the basis,
        or leave it out entirely. Default is "after".
    priors : dict[str, Prior]
        Prior for the sigma_before, sigma_after, a_before, and a_after parameters.
    prefix : str
        Prefix for the parameters.
    """

    lookup_name = "asymmetric_gaussian"

    def __init__(
        self,
        event_in: Literal["before", "after", "exclude"] = "after",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.event_in = event_in

    def function(
        self,
        x: XTensorLike,
        sigma_before: XTensorLike,
        sigma_after: XTensorLike,
        a_after: XTensorLike,
        *,
        dim: str | None = None,
    ) -> XTensorVariable:
        """Asymmetric Gaussian bump function."""
        x = as_xtensor(x)

        match self.event_in:
            case "before":
                indicator_before = ptx.math.cast(x <= 0, "float32")
                indicator_after = ptx.math.cast(x > 0, "float32")
            case "after":
                indicator_before = ptx.math.cast(x < 0, "float32")
                indicator_after = ptx.math.cast(x >= 0, "float32")
            case "exclude":
                indicator_before = ptx.math.cast(x < 0, "float32")
                indicator_after = ptx.math.cast(x > 0, "float32")
            case _:
                raise ValueError(f"Invalid event_in: {self.event_in}")

        rv_before = pmd.Normal.dist(mu=0.0, sigma=sigma_before)
        rv_after = pmd.Normal.dist(mu=0.0, sigma=sigma_after)

        y_before = ptx.math.switch(
            indicator_before,
            ptx.math.exp(logp(rv_before, x)),
            0,
        )
        y_after = ptx.math.switch(
            indicator_after,
            ptx.math.exp(logp(rv_after, x)) * a_after,
            0,
        )

        return y_before + y_after

    def to_dict(self) -> dict:
        """Convert the asymmetric Gaussian basis to a dictionary."""
        return {
            **super().to_dict(),
            "event_in": self.event_in,
        }

    default_priors = {
        "sigma_before": Prior("Gamma", mu=3, sigma=1),
        "sigma_after": Prior("Gamma", mu=7, sigma=2),
        "a_after": Prior("Normal", mu=1, sigma=0.5),
    }


def days_from_reference(
    dates: pd.Series | pd.DatetimeIndex,
    reference_date: str | pd.Timestamp,
) -> npt.NDArray[np.int64]:
    """Calculate the difference in days between dates and a reference date.

    Parameters
    ----------
    dates : pd.Series | pd.DatetimeIndex
        Dates to calculate the difference from the reference date.
    reference_date : str | pd.Timestamp
        Reference date.

    Returns
    -------
    np.ndarray
        Difference in days between dates and the reference date.

    """
    reference_date = cast(pd.Timestamp, pd.to_datetime(reference_date))
    dates = pd.to_datetime(dates)

    diff = dates - reference_date

    if isinstance(diff, pd.Series):
        diff = diff.dt  # type: ignore

    return diff.days.to_numpy()  # type: ignore
