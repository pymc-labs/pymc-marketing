from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy.typing as npt
import pandas as pd
import pymc as pm
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pymmmc.transformers import geometric_adstock_vectorized, logistic_saturation
from pymmmc.utils import generate_fourier_modes


@dataclass
class DataContainer:
    """Data container for the model features and transformations."""

    raw_data: pd.DataFrame
    transformer: Optional[BaseEstimator] = None  # ? `Pipeline` types?
    data: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        if self.transformer is not None:
            self.transformer.fit(self.raw_data)
            self.data = pd.DataFrame(
                data=self.transformer.transform(self.raw_data),
                columns=self.raw_data.columns,
            )
        else:
            self.data = self.raw_data.copy()


class Model:
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        date_col: str,
        media_columns: List[str],
        control_columns_continuous: Optional[List[str]],
        control_columns_categorical: Optional[List[str]],
        model_type: str = "adstock_saturation",  # or "saturation" or "adstock" or ...
        fourier_modes: int = 3,
    ) -> None:
        self.data = data
        self.target = target
        self.date_col = date_col
        self.media_columns = media_columns
        self.control_continuous_columns = control_columns_continuous
        self.control_categorical_columns = control_columns_categorical
        self.model_type = model_type
        self.fourier_modes = fourier_modes

    def _prepare_data(
        self,
    ) -> Tuple[
        npt.NDArray,
        DataContainer,
        DataContainer,
        Optional[DataContainer],
        Optional[DataContainer],
        Optional[DataContainer],
    ]:
        """Data processing step. We probably want to modularize this method."""
        # extract data
        y: npt.NDArray = self.data[self.target].to_numpy()
        date: npt.NDArray = self.data[self.date_col].to_numpy()
        media_df: pd.DataFrame = self.data[self.media_columns]
        control_continuous_df: pd.DataFrame = (
            self.data[self.control_continuous_columns]
            if self.control_continuous_columns
            else None
        )
        control_categorical_df: pd.DataFrame = (
            self.data[self.control_categorical_columns]
            if self.control_categorical_columns
            else None
        )
        # parse as data containers
        y_data: DataContainer = DataContainer(raw_data=y, transformer=StandardScaler())
        media_data: DataContainer = DataContainer(
            raw_data=media_df, transformer=MinMaxScaler()
        )
        control_continuous_data: Optional[DataContainer] = (
            DataContainer(raw_data=control_continuous_df, transformer=StandardScaler())
            if control_continuous_df
            else None
        )
        control_categorical_data: Optional[DataContainer] = (
            DataContainer(raw_data=control_categorical_df, transformer=None)
            if control_categorical_df
            else None
        )
        fourier_features: Optional[pd.DataFrame] = (
            generate_fourier_modes(
                periods=self.data[self.date_col].dt.dayofyear / 365.25,
                n_order=self.fourier_models,
            )
            if self.fourier_models
            else None
        )
        fourier_features_data: Optional[DataContainer] = (
            DataContainer(raw_data=fourier_features) if fourier_features else None
        )

        return (
            date,
            y_data,
            media_data,
            control_continuous_data,
            control_categorical_data,
            fourier_features_data,
        )

    def _build_model(self) -> pm.Model:
        (
            date,
            y_data,
            media_data,
            control_continuous_data,
            control_categorical_data,
            fourier_features_data,
        ) = self._prepare_data()

        coords: Dict[str : Union[npt.NDArray, pd.DataFrame]] = {
            "date_week": date,
            "channel": media_data.data,
        }
        if self.control_continuous_columns:
            coords["control"] = control_continuous_data.data
        if self.control_categorical_columns:
            coords["control_categorical"] = control_categorical_data.data
        if self.fourier_modes:
            coords["fourier_mode"] = fourier_features_data.data

        with pm.Model(coords=coords) as model:

            # --- Priors ---
            intercept = pm.Normal(name="intercept", mu=0, sigma=2)
            # coefficients marketing channels
            beta = pm.HalfNormal(name="beta", sigma=2, dims="channel")
            # adstock parameter
            alpha = pm.Beta(name="alpha", alpha=1, beta=3, dims="channel")
            # saturation parameter
            lam = pm.Gamma(name="lam", alpha=3, beta=1, dims="channel")

            # coefficients control variables
            if self.control_continuous_columns:
                gamma_control_continuous = pm.Laplace(
                    name="gamma_control_continuous", mu=0, b=1, dims="control"
                )
            if self.control_categorical_columns:
                gamma_control_categorical = pm.Laplace(
                    name="gamma_control_categorical", mu=0, b=1, dims="control"
                )
            # coefficients fourier modes
            if self.fourier_modes:
                gamma_fourier = pm.Laplace(
                    name="gamma_fourier", mu=0, b=1, dims="fourier_mode"
                )
            # likelihood standard deviation
            sigma = pm.HalfNormal(name="sigma", sigma=2)
            # degrees of freedom
            nu = pm.Gamma(name="nu", alpha=25, beta=2)

            # --- Model Parametrization ---
            # marketing channel effects
            # # saturated-adstock
            # TODO: use the appropriate composition defined by the attribute `model_type`. # noqa E501
            channels_saturated = pm.Deterministic(
                name="channels_saturated",
                var=logistic_saturation(x=media_data.data, lam=lam),
                dims=("date_week", "channel"),
            )
            channels_saturated_adstock = pm.Deterministic(
                name="channels_saturated_adstock",
                var=geometric_adstock_vectorized(
                    x=channels_saturated,
                    alpha=alpha,
                    l_max=12,
                    normalize=True,
                ),
                dims=("date_week", "channel"),
            )
            channels_effects = pm.Deterministic(
                name="channels_effects",
                var=pm.math.dot(channels_saturated_adstock, beta),
                dims=("date_week"),
            )
            additive_effects = [intercept, channels_effects]

            # control variables effect
            if self.control_continuous_columns:
                control_continuous_effects = pm.Deterministic(
                    name="control_effects",
                    var=pm.math.dot(
                        control_continuous_data.data, gamma_control_continuous
                    ),
                    dims="date_week",
                )
                additive_effects.append(control_continuous_effects)

            if self.control_categorical_columns:
                control_categorical_effects = pm.Deterministic(
                    name="control_effects",
                    var=pm.math.dot(
                        control_categorical_data.data, gamma_control_categorical
                    ),
                    dims="date_week",
                )
                additive_effects.append(control_categorical_effects)

            # fourier modes effect
            if self.fourier_modes:
                fourier_effects = pm.Deterministic(
                    name="fourier_effects",
                    var=pm.math.dot(fourier_features_data, gamma_fourier),
                    dims="date_week",
                )
                additive_effects.append(fourier_effects)

            mu = pm.Deterministic(
                name="mu",
                var=sum(additive_effects),
                dims="date_week",
            )

            # --- Likelihood ---
            likelihood = pm.StudentT(  # noqa F841
                "likelihood",
                mu=mu,
                nu=nu,
                sigma=sigma,
                dims="date",
                observed=y_data.data,
            )
        return model

    def prior_predictive(self) -> None:
        pass

    def plot_prior_predictive(self) -> None:
        pass

    def fit(self) -> None:
        pass
