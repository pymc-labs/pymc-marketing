#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Frequency-Reach additive effect.

Implements an additive mu effect based on pre-computed (or provided) empirical
frequency / reach observations per channel over time, following the Google
Meridian style reach-frequency modelling guidance:
https://developers.google.com/meridian/docs/advanced-modeling/reach-frequency

The objective is to transform raw marketing activity that is represented in
terms of observed average frequency (times a reached individual is exposed)
and reach (portion of the target population reached) into an effective
"exposure pressure" term which is then passed through the existing adstock
and saturation pipeline (re-using the same transformation classes already
available to standard channel contributions) and added additively to the
model mean (mu).

Data Requirements
-----------------
`df_frequency_reach` must contain at least the following columns:
    * date: (datetime-like) date of the observation.
    * channel: str categorical identifying the channel (aligned with MMM coords)
    * frequency: numeric (>=0) average frequency among the reached population.
    * reach: numeric in [0, 1] proportion of the target population reached.

Optionally additional dimensions aligned with `mmm.dims` can be included
and will be preserved (e.g. geo). They must exactly match those dims' names.

Transformation Logic (Meridian-style default)
-------------------------------------------
1. Register two raw tensors: reach_raw and frequency_raw with dims
    (date, *mmm.dims, channel).
2. Apply the provided saturation transformation ONLY to frequency_raw to get
    frequency_sat (reach remains linear / unsaturated).
3. Form effective_exposure_raw = reach_raw * frequency_sat (element-wise).
4. Apply the adstock transformation to effective_exposure_raw producing
    effective_exposure_adstocked.
5. Draw per-channel scaling coefficients beta[channel] and compute
    channel_contribution = beta * effective_exposure_adstocked.
6. Aggregate over channel to obtain total_effect added to model mean.
7. Expose intermediate deterministics for diagnostics:
    frequency_sat, effective_exposure_raw, effective_exposure_adstocked,
    channel_contribution, total_effect.

Assumptions
-----------
* Frequencies and reaches are already on the relevant date granularity of the model.
* Reach is a proportion in [0,1]; frequency is >=0.
* Missing combinations (date, channel, extra dims) are zero-filled internally.
* Saturation acts only on frequency to avoid stacking nonlinearities on both
    multiplicative terms (improves identifiability of beta).

Edge Cases & Validation
-----------------------
* Negative frequency or reach outside [0,1] raises ValueError.
* Duplicate rows for the same (date, dims, channel) are aggregated (mean).
* If channels present in the DataFrame are not a subset of the MMM channel
  coordinate, a ValueError is raised.

Extensibility
-------------
* Alternate effective exposure formulas (e.g. reach transform variants).
* Hierarchical priors for beta across channels / geos.
* Alternative adstock kernels via the injected transformation object.
* Additional nonlinear transforms on reach if strongly justified (not default).
"""

from typing import Any

import pandas as pd
import pymc as pm
import xarray as xr
from pydantic import BaseModel, InstanceOf
from pymc_extras.prior import create_dim_handler

from pymc_marketing.mmm.additive_effect import Model
from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation


class FrequencyReachAdditiveEffect(BaseModel):
    """Additive mu effect from frequency & reach observations.

    Parameters
    ----------
    df_frequency_reach : pd.DataFrame
        Long format DataFrame with columns at least: date, channel, frequency, reach.
        Additional columns matching `mmm.dims` (e.g. geo) are optional.
    saturation : SaturationTransformation
        Saturation transformation applied after adstock.
    adstock : AdstockTransformation
        Adstock transformation applied first to the effective exposure signal.
    prefix : str, default "frequency_reach"
        Variable name prefix for PyMC random/deterministic vars.
    date_dim_name : str, default "date"
        Name of the date coordinate in the target model.
    adstock_first : bool, default True (deprecated / enforced True)
        Kept for backward compatibility with earlier drafts; the current
        implementation always performs: saturation(frequency) -> multiply by reach -> adstock.
    channel_coord_name : str, default "channel"
        Name of the channel coordinate used by the parent model.
    """

    df_frequency_reach: InstanceOf[pd.DataFrame]
    saturation: InstanceOf[SaturationTransformation]
    adstock: InstanceOf[AdstockTransformation]
    prefix: str = "frequency_reach"
    date_dim_name: str = "date"
    adstock_first: bool = True  # Enforced True (see model_post_init)
    channel_coord_name: str = "channel"

    # Internal cached shapes (populated at create_data)
    _mmm_dims: tuple[str, ...] | None = None

    def model_post_init(self, context: Any, /) -> None:  # type: ignore[override]
        """Model post initialization for a Pydantic model."""
        required = {self.date_dim_name, "frequency", "reach", self.channel_coord_name}
        missing = required.difference(self.df_frequency_reach.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in df_frequency_reach: {missing}"
            )

        # Basic validation of ranges
        if (
            self.df_frequency_reach["reach"].lt(0)
            | self.df_frequency_reach["reach"].gt(1)
        ).any():
            raise ValueError("Reach must be within [0, 1].")
        if (self.df_frequency_reach["frequency"].lt(0)).any():
            raise ValueError("Frequency must be non-negative.")

        # Ensure datetime
        self.df_frequency_reach[self.date_dim_name] = pd.to_datetime(
            self.df_frequency_reach[self.date_dim_name]
        )

        # Assign unique prefixes to the internal transformations so their
        # parameter variable names don't clash with the main model's
        # transformations (e.g. adstock_alpha, saturation_beta, etc.).
        # This mirrors the usage of prefixes in other effects (e.g. FourierEffect)
        # without cloning the transformation objects.
        if hasattr(self.adstock, "prefix"):
            self.adstock.prefix = f"{self.prefix}_adstock"
        if hasattr(self.saturation, "prefix"):
            self.saturation.prefix = f"{self.prefix}_saturation"

        # Enforce adstock_first True (we currently rely on that ordering for interpretation)
        if not self.adstock_first:
            raise ValueError(
                "FrequencyReachAdditiveEffect currently requires adstock_first=True. "
                "Support for saturation-first ordering is not implemented."
            )

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------
    def create_data(self, mmm: Model) -> None:
        """Register pm.Data nodes required for the effect.

        Creates two pm.Data tensors:
            * `<prefix>_reach_raw`
            * `<prefix>_frequency_raw`
        each with dims (date, *mmm.dims, channel) ready for transformation.
        """
        self._mmm_dims = mmm.dims
        freq_np, reach_np = self._build_raw_arrays(self.df_frequency_reach.copy(), mmm)
        pm.Data(
            f"{self.prefix}_frequency_raw",
            freq_np,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )
        pm.Data(
            f"{self.prefix}_reach_raw",
            reach_np,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )

    def create_effect(self, mmm: Model):
        """Create transformed contribution (Meridian pipeline) and return aggregate effect.

        Steps inside the model graph:
            frequency_sat = saturation(frequency_raw)
            effective_exposure_raw = reach_raw * frequency_sat
            effective_exposure_adstocked = adstock(effective_exposure_raw)
            channel_contribution = beta * effective_exposure_adstocked
            total_effect = sum_channel channel_contribution

        Returns a tensor with dims (date, *mmm.dims) to be added to mu.
        """
        model = mmm.model
        freq_raw = model[f"{self.prefix}_frequency_raw"]
        reach_raw = model[f"{self.prefix}_reach_raw"]

        # 1. Saturate frequency ONLY
        frequency_sat = self.saturation.apply(x=freq_raw, dims=self.channel_coord_name)
        pm.Deterministic(
            f"{self.prefix}_frequency_sat",
            frequency_sat,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )

        # 2. Element-wise product with linear reach
        effective_exposure_raw = reach_raw * frequency_sat
        pm.Deterministic(
            f"{self.prefix}_effective_exposure_raw",
            effective_exposure_raw,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )

        # 3. Adstock
        effective_exposure_adstocked = self.adstock.apply(
            x=effective_exposure_raw, dims=self.channel_coord_name
        )
        pm.Deterministic(
            f"{self.prefix}_effective_exposure_adstocked",
            effective_exposure_adstocked,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )

        # 4. Per-channel beta scaling
        beta = pm.Normal(
            f"{self.prefix}_beta",
            mu=0.0,
            sigma=1.0,
            dims=(self.channel_coord_name,),
        )
        channel_contribution = pm.Deterministic(
            f"{self.prefix}_channel_contribution",
            beta * effective_exposure_adstocked,
            dims=(self.date_dim_name, *mmm.dims, self.channel_coord_name),
        )

        # 5. Aggregate over channel
        total_effect = pm.Deterministic(
            f"{self.prefix}_total_effect",
            channel_contribution.sum(axis=-1),
            dims=(self.date_dim_name, *mmm.dims),
        )

        dim_handler = create_dim_handler((self.date_dim_name, *mmm.dims))
        return dim_handler(total_effect, (self.date_dim_name, *mmm.dims))

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Update reach & frequency raw data for prediction dates.

        Extends the registered pm.Data nodes `<prefix>_frequency_raw` and
        `<prefix>_reach_raw` to any new dates in the model coordinates, filling
        unseen future dates with zeros. Does not accept a modified DataFrame;
        construct a new effect instance if the historical data themselves change.
        """
        raw_dates = model.coords.get(self.date_dim_name)
        if raw_dates is None:
            raise ValueError(
                f"Model missing '{self.date_dim_name}' coordinate during set_data."
            )
        new_dates = pd.to_datetime(list(raw_dates))
        df_extended = self.df_frequency_reach.copy()
        max_original_date = df_extended[self.date_dim_name].max()

        future_mask = new_dates > max_original_date
        if future_mask.any():
            add_rows = []
            channels = model.coords.get(self.channel_coord_name)
            if channels is None:
                raise ValueError(
                    f"Model missing '{self.channel_coord_name}' coordinate during set_data."  # pragma: no cover
                )
            if self._mmm_dims is not None:
                # mypy/pydantic may treat _mmm_dims as Optional; runtime guard above ensures safety
                dims_product = [list(model.coords[d]) for d in self._mmm_dims]  # type: ignore[arg-type]
            else:
                dims_product = []
            from itertools import product

            # Prepare iterable of dimension combinations (empty tuple if no extra dims)
            dim_combos = product(*dims_product) if dims_product else [()]
            for date in new_dates[future_mask]:
                for combo in dim_combos:
                    for ch in channels:
                        row = {
                            self.date_dim_name: date,
                            self.channel_coord_name: ch,
                            "frequency": 0.0,
                            "reach": 0.0,
                        }
                        if self._mmm_dims:
                            for dim_name, dim_val in zip(
                                self._mmm_dims, combo, strict=False
                            ):
                                row[dim_name] = dim_val
                        add_rows.append(row)
            if add_rows:
                df_extended = pd.concat(
                    [df_extended, pd.DataFrame(add_rows)], ignore_index=True
                )

        # Rebuild raw arrays directly (no temporary instance) and update
        freq_np, reach_np = self._build_raw_arrays(df_extended, mmm)
        pm.set_data(
            {
                f"{self.prefix}_frequency_raw": freq_np,
                f"{self.prefix}_reach_raw": reach_np,
            },
            model=model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_raw_arrays(self, df: pd.DataFrame, mmm: Model) -> tuple[Any, Any]:
        """Return reshaped frequency & reach arrays aligned to model coords.

        Shared by create_data (initial build) and set_data (future extension).
        """
        model = mmm.model
        channel_coord = model.coords.get(self.channel_coord_name)
        if channel_coord is None:
            raise ValueError(
                f"Parent model has no '{self.channel_coord_name}' coordinate."
            )

        # Validate channel subset (only during create; harmless on updates)
        unique_channels = df[self.channel_coord_name].unique()
        if not set(unique_channels).issubset(set(channel_coord)):
            extra = set(unique_channels).difference(set(channel_coord))
            raise ValueError(
                f"Channels {extra} in frequency-reach data not present in model coord."
            )

        raw_dates = model.coords.get(self.date_dim_name)
        if raw_dates is None:
            raise ValueError(f"Model missing '{self.date_dim_name}' coordinate.")
        model_dates = pd.to_datetime(list(raw_dates))

        # Keep only allowed dims/columns
        allowed_dims = {
            self.date_dim_name,
            self.channel_coord_name,
            "frequency",
            "reach",
            *mmm.dims,
        }
        drop_cols = [c for c in df.columns if c not in allowed_dims]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        group_keys = [self.date_dim_name, *mmm.dims, self.channel_coord_name]
        df = (
            df.groupby(group_keys, dropna=False)[["frequency", "reach"]]
            .mean()
            .reset_index()
        )

        # Cartesian product index
        iterables: list[list[Any]] = [list(model_dates)]  # type: ignore[var-annotated]
        for dim in mmm.dims:
            coord_vals = model.coords.get(dim)
            if coord_vals is None:
                raise ValueError(
                    f"Model missing dim coordinate '{dim}' required by MMM dims"
                )
            iterables.append(list(coord_vals))
        iterables.append(list(channel_coord))
        full_index = pd.MultiIndex.from_product(
            iterables, names=[self.date_dim_name, *mmm.dims, self.channel_coord_name]
        )

        df = df.set_index(group_keys).reindex(full_index)
        df[["frequency", "reach"]] = df[["frequency", "reach"]].fillna(0.0)
        df = df.reset_index()

        df = df.set_index([self.date_dim_name, *mmm.dims, self.channel_coord_name])
        freq_wide = (
            df["frequency"]
            .unstack(self.channel_coord_name)
            .reindex(columns=channel_coord)
        )
        reach_wide = (
            df["reach"].unstack(self.channel_coord_name).reindex(columns=channel_coord)
        )

        n_dates = len(model_dates)
        dim_sizes = []
        for d in mmm.dims:
            coord_vals = model.coords.get(d)
            if coord_vals is None:
                raise ValueError(
                    f"Model missing coordinate for dim '{d}' while reshaping exposure tensor."
                )
            dim_sizes.append(len(list(coord_vals)))
        n_channels = len(channel_coord)

        expected_rows = n_dates
        for s in dim_sizes:
            expected_rows *= s
        if freq_wide.shape[0] != expected_rows or reach_wide.shape[0] != expected_rows:
            raise ValueError(
                "Internal shape mismatch while constructing reach/frequency tensors: "
                f"expected {expected_rows} rows, found freq={freq_wide.shape[0]}, reach={reach_wide.shape[0]}."
            )

        freq_np = (
            freq_wide.to_numpy()
            .reshape(n_dates, *dim_sizes, n_channels)
            .astype("float32")
        )
        reach_np = (
            reach_wide.to_numpy()
            .reshape(n_dates, *dim_sizes, n_channels)
            .astype("float32")
        )
        return freq_np, reach_np


__all__ = ["FrequencyReachAdditiveEffect"]
