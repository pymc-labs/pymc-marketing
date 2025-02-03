from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pydantic import Field, InstanceOf, validate_call

from pymc_marketing.deserialize import deserialize, register_deserialization
from pymc_marketing.mmm.components.base import Transformation, create_registration_meta
from pymc_marketing.prior import Prior, create_dim_handler

BASIS_TRANSFORMATIONS = {}
BasisMeta = create_registration_meta(BASIS_TRANSFORMATIONS)


class Basis(Transformation, metaclass=BasisMeta):
    prefix: str = "basis"
    lookup_name: str

    @validate_call
    def sample_curve(
        self,
        parameters: InstanceOf[xr.Dataset] = Field(
            ..., description="Parameters of the saturation transformation."
        ),
        days: int = Field(0, ge=0, description="Minimum number of days."),
    ) -> xr.DataArray:
        """Sample the curve of the saturation transformation given parameters.

        Parameters
        ----------
        parameters : xr.Dataset
            Dataset with the parameters of the saturation transformation.
        max_value : float, optional
            Maximum value of the curve, by default 1.0.

        Returns
        -------
        xr.DataArray
            Curve of the saturation transformation.

        """
        x = np.linspace(-days, days, 100)

        coords = {
            "x": x,
        }

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


@dataclass
class EventEffect:
    basis: Basis
    effect_size: Prior
    dims: tuple[str, ...]

    def apply(self, X, name: str = "event"):
        """Apply the event effect to the data."""
        dim_handler = create_dim_handler(("x", *self.dims))
        return self.basis.apply(X, dims=self.dims) * dim_handler(
            self.effect_size.create_variable(f"{name}_effect_size"),
            self.effect_size.dims,
        )

    def to_dict(self):
        return {
            "class": "EventEffect",
            "data": {
                "basis": self.basis.to_dict(),
                "effect_size": self.effect_size.to_dict(),
                "dims": self.dims,
            },
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            basis=deserialize(data["basis"]),
            effect_size=deserialize(data["effect_size"]),
            dims=data["dims"],
        )


def _is_event_effect(data):
    return data["class"] == "EventEffect"


register_deserialization(
    is_type=_is_event_effect,
    deserialize=lambda data: EventEffect.from_dict(data["data"]),
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from pymc_marketing.plot import plot_curve

    class GaussianBasis(Basis):
        lookup_name = "gaussian"

        def function(self, x, sigma):
            return pm.math.exp(-0.5 * (x / sigma) ** 2)

        default_priors = {
            "sigma": Prior("Gamma", mu=7, sigma=1),
        }

    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=[4, 7, 10], sigma=1, dims="event"),
        },
    )
    coords = {"event": ["NYE", "Grand Opening Game Show", "Super Bowl"]}
    prior = gaussian.sample_prior(coords=coords)
    curve = gaussian.sample_curve(prior, days=21)

    fig, axes = gaussian.plot_curve(curve, same_axes=True)
    fig.suptitle("Gaussian Basis")
    plt.savefig("gaussian-basis")
    plt.close()

    df_events = pd.DataFrame(
        {
            "event": ["first", "second"],
            "start_date": pd.to_datetime(["2023-01-01", "2023-01-20"]),
            "end_date": pd.to_datetime(["2023-01-02", "2023-01-25"]),
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

        s_ref = difference_in_days(model_dates, start_dates)
        e_ref = difference_in_days(model_dates, end_dates)

        return np.where(
            (s_ref >= 0) & (e_ref <= 0),
            0,
            np.where(np.abs(s_ref) < np.abs(e_ref), s_ref, e_ref),
        )

    gaussian = GaussianBasis(
        priors={
            "sigma": Prior("Gamma", mu=7, sigma=1, dims="event"),
        }
    )
    effect_size = Prior("Normal", mu=1, sigma=1, dims="event")
    effect = EventEffect(basis=gaussian, effect_size=effect_size, dims=("event",))

    dates = pd.date_range("2022-12-01", periods=3 * 31, freq="D")

    X = create_basis_matrix(df_events, model_dates=dates)

    coords = {"date": dates, "event": df_events["event"].to_numpy()}
    with pm.Model(coords=coords) as model:
        pm.Deterministic("effect", effect.apply(X), dims=("date", "event"))

        idata = pm.sample_prior_predictive()

    fig, axes = idata.prior.effect.pipe(
        plot_curve,
        {"date"},
        subplot_kwargs={"ncols": 1},
    )
    fig.suptitle("Gaussian Event Effect")
    plt.savefig("gaussian-event")
    plt.close()
