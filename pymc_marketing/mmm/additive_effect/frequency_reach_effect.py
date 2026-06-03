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
``df_frequency_reach`` must contain at least the following columns:

* ``date``: datetime-like date of the observation.
* ``channel``: str categorical identifying the R&F channel.
* ``frequency``: numeric (>=0) average frequency among the reached population.
* ``reach``: numeric in [0, 1] proportion of the target population reached.

Optionally additional dimensions aligned with ``mmm.dims`` can be included
and will be preserved (e.g. geo). They must exactly match those dims' names.

R&F channels are independent of the standard MMM ``channel_columns``. They use
a dedicated ``rf_channel`` coordinate in the PyMC model. R&F is an *alternative*
representation of media activity for those channels — the reach and frequency
data replaces impressions or spend, not supplements it.  Any channel name that
appears in ``df_frequency_reach`` **must not** also appear in
``channel_columns`` of the parent MMM: ``create_data`` raises a ``ValueError``
if this constraint is violated, mirroring Meridian's own
``InputData._validate_media_channels`` check.

Transformation Logic (Meridian-style)
--------------------------------------
1. Register two raw tensors: ``{prefix}_reach_raw`` and
   ``{prefix}_frequency_raw`` with dims ``(date, *mmm.dims, rf_channel)``.
2. Apply the provided saturation transformation ONLY to ``frequency_raw`` to
   get ``frequency_sat`` (reach remains linear / unsaturated).
3. Form ``effective_exposure_raw = reach_raw * frequency_sat`` (element-wise).
4. Apply the adstock transformation to ``effective_exposure_raw`` producing
   ``effective_exposure_adstocked``.
5. Draw per-channel scaling coefficients ``beta[rf_channel]`` and compute
   ``channel_contribution = beta * effective_exposure_adstocked``.
6. Aggregate over rf_channel to obtain ``total_effect`` added to model mean.
7. Expose intermediate deterministics for diagnostics:
   ``frequency_sat``, ``effective_exposure_raw``,
   ``effective_exposure_adstocked``, ``channel_contribution``,
   ``total_effect``.

Budget Optimization
--------------------
When ``cost_per_unit`` is provided on this effect, the
:class:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer` can optimize
spend across R&F channels jointly with standard media channels.

The conversion chain per channel ``i`` during optimization is:

.. code-block:: text

    spend_i → ÷ cost_per_unit_i → impressions_i
    impressions_i = reach_i × frequency_i*
    frequency_i* = assumed_frequency (fixed; historical median if not set)
    reach_i = impressions_i / frequency_i*

This follows the same approach as Meridian: frequency is fixed during budget
optimization (either at a user-supplied value or at the historical median), and
only the total spend per channel is optimized.

Assumptions
-----------
* Frequencies and reaches are already on the relevant date granularity of the model.
* Reach is a proportion in [0, 1]; frequency is >= 0.
* Missing combinations (date, channel, extra dims) are zero-filled internally.
* Saturation acts only on frequency to avoid stacking nonlinearities on both
  multiplicative terms (improves identifiability of beta).
* R&F channels are disjoint from ``channel_columns`` of the parent MMM.

Edge Cases & Validation
-----------------------
* Negative frequency or reach outside [0, 1] raises ``ValueError``.
* Duplicate rows for the same (date, dims, channel) are aggregated (mean).
* ``cost_per_unit``, if provided, must have a ``date`` column and one column
  per R&F channel, with all-positive values.

References
----------
Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017).
Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
Google Inc.

Guo, R., Chan, D., Koehler, J., Jin, Y., Wang, Y., & Sun, Y. (2021).
Bayesian Hierarchical Media Mix Model Incorporating Reach and Frequency Data.
https://research.google/pubs/bayesian-hierarchical-media-mix-model-incorporating-reach-and-frequency-data/
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import BaseModel, Field, InstanceOf, model_validator

from pymc_marketing.mmm.additive_effect.additive_effect import Model
from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import HillShapeSaturation
from pymc_marketing.prior import Prior

# The PyMC coordinate name for R&F channels — kept separate from the standard
# "channel" coordinate to guarantee disjointness with channel_columns.
RF_CHANNEL_COORD = "rf_channel"


class FrequencyReachAdditiveEffect(BaseModel):
    """Additive mu effect from frequency & reach observations.

    Follows the Meridian-style reach-frequency modelling approach:
    saturation is applied to frequency only, reach enters linearly, and the
    product is passed through adstock before being scaled by a per-channel
    beta coefficient.

    R&F channels live in a dedicated ``rf_channel`` coordinate and are fully
    independent of the standard MMM ``channel_columns``.

    Parameters
    ----------
    df_frequency_reach : pd.DataFrame
        Long-format DataFrame with columns at least: ``date``, ``channel``,
        ``frequency``, ``reach``.  Additional columns matching ``mmm.dims``
        (e.g. ``geo``) are optional and will be preserved.
    saturation : HillShapeSaturation
        Shape-only Hill saturation applied to frequency.  Must not contain an
        internal amplitude parameter to keep the model identifiable (that role
        is played by ``beta``).
    adstock : AdstockTransformation
        Adstock kernel applied to the effective-exposure signal
        (``reach × saturated_frequency``).
    beta_prior : Prior, optional
        Prior for the per-channel scaling coefficient ``beta``.
        Defaults to ``HalfNormal(sigma=1)``, which enforces non-negative
        contributions (semantically appropriate for R&F channels). Any
        ``Prior`` instance is accepted; dims must not be set — they are
        assigned automatically to ``"rf_channel"`` at model build time.
    cost_per_unit : pd.DataFrame, optional
        Cost per impression for each R&F channel, optionally varying by date.
        Wide-format DataFrame with a ``date`` column and one column per channel
        name matching those in ``df_frequency_reach``.  All values must be
        positive.  Required when budget optimization is used.
    assumed_frequency : float or dict[str, float], optional
        Fixed average-frequency assumption used during budget optimization to
        split total impressions into reach and frequency.  A scalar applies to
        all channels; a dict maps channel names to per-channel values.  When
        ``None`` (default), the historical per-channel median frequency is used.
    prefix : str, optional
        Variable-name prefix for all PyMC random and deterministic variables
        created by this effect.  Defaults to ``"frequency_reach"``.
    date_column : str, optional
        Name of the date column in ``df_frequency_reach``.
        Defaults to ``"date"``.
    channel_column : str, optional
        Name of the channel column in ``df_frequency_reach``.
        Defaults to ``"channel"``.

    Notes
    -----
    The PyMC coordinate for R&F channels is always ``"rf_channel"`` regardless
    of ``channel_column``.  This ensures that R&F channel names cannot collide
    with standard media channel coordinates.
    """

    model_config = {"arbitrary_types_allowed": True}

    df_frequency_reach: InstanceOf[pd.DataFrame]
    saturation: InstanceOf[HillShapeSaturation]
    adstock: InstanceOf[AdstockTransformation]
    beta_prior: Prior = Field(default_factory=lambda: Prior("HalfNormal", sigma=1))
    cost_per_unit: InstanceOf[pd.DataFrame] | None = None
    assumed_frequency: float | dict[str, float] | None = None
    prefix: str = "frequency_reach"
    date_column: str = "date"
    channel_column: str = "channel"

    # Internal cached shapes (populated at create_data)
    _mmm_dims: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> FrequencyReachAdditiveEffect:
        """Validate DataFrame contents and cost_per_unit."""
        df = self.df_frequency_reach

        # Required columns
        required = {self.date_column, "frequency", "reach", self.channel_column}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in df_frequency_reach: {missing}"
            )

        # Range checks
        if (df["reach"].lt(0) | df["reach"].gt(1)).any():
            raise ValueError("Reach must be within [0, 1].")
        if df["frequency"].lt(0).any():
            raise ValueError("Frequency must be non-negative.")

        # Ensure datetime
        self.df_frequency_reach[self.date_column] = pd.to_datetime(df[self.date_column])

        # Validate cost_per_unit DataFrame if provided
        if self.cost_per_unit is not None:
            cpu = self.cost_per_unit
            if "date" not in cpu.columns:
                raise ValueError("cost_per_unit must have a 'date' column.")
            rf_channels = set(df[self.channel_column].unique())
            cpu_channels = set(cpu.columns) - {"date"}
            unknown = cpu_channels - rf_channels
            if unknown:
                raise ValueError(
                    f"cost_per_unit contains channels {unknown} not found in "
                    "df_frequency_reach."
                )
            value_cols = [c for c in cpu.columns if c != "date"]
            if (cpu[value_cols] <= 0).any().any():
                raise ValueError("cost_per_unit values must be positive.")

        # Assign unique prefixes to internal transformations to avoid name clashes
        if hasattr(self.adstock, "prefix"):
            self.adstock.prefix = f"{self.prefix}_adstock"
        if hasattr(self.saturation, "prefix"):
            self.saturation.prefix = f"{self.prefix}_saturation"

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rf_channels(self) -> list[str]:
        """Sorted list of R&F channel names from the DataFrame."""
        return sorted(self.df_frequency_reach[self.channel_column].unique().tolist())

    @property
    def cost_per_unit_xarray(self) -> xr.DataArray | None:
        """Parse ``cost_per_unit`` DataFrame into a DataArray.

        Returns
        -------
        xr.DataArray or None
            DataArray with dims ``(date, rf_channel)`` if ``cost_per_unit`` is
            set, otherwise ``None``.
        """
        if self.cost_per_unit is None:
            return None

        cpu = self.cost_per_unit.copy()
        cpu["date"] = pd.to_datetime(cpu["date"])
        channels = self.rf_channels
        cpu = cpu.set_index("date")[channels]
        return xr.DataArray(
            cpu.values.astype("float64"),
            dims=["date", RF_CHANNEL_COORD],
            coords={"date": cpu.index.values, RF_CHANNEL_COORD: channels},
        )

    def get_assumed_frequency_array(self) -> np.ndarray:
        """Return a per-channel frequency array for use during optimization.

        Returns
        -------
        np.ndarray
            Shape ``(n_rf_channels,)`` — the assumed frequency per channel, in
            the same order as :attr:`rf_channels`.
        """
        channels = self.rf_channels
        if self.assumed_frequency is None:
            # Fall back to historical per-channel median
            return np.array(
                [
                    float(
                        self.df_frequency_reach.loc[
                            self.df_frequency_reach[self.channel_column] == ch,
                            "frequency",
                        ].median()
                    )
                    for ch in channels
                ],
                dtype="float64",
            )
        if isinstance(self.assumed_frequency, dict):
            missing = set(channels) - set(self.assumed_frequency)
            if missing:
                raise ValueError(
                    f"assumed_frequency dict is missing channels: {missing}"
                )
            return np.array(
                [self.assumed_frequency[ch] for ch in channels], dtype="float64"
            )
        # scalar
        return np.full(len(channels), float(self.assumed_frequency), dtype="float64")

    # ------------------------------------------------------------------
    # MuEffect protocol methods
    # ------------------------------------------------------------------

    def create_data(self, mmm: Model) -> None:
        """Register pm.Data nodes required for the effect.

        Creates two pm.Data tensors:

        * ``{prefix}_frequency_raw`` with dims ``(date, *mmm.dims, rf_channel)``
        * ``{prefix}_reach_raw``     with dims ``(date, *mmm.dims, rf_channel)``

        Also adds an ``rf_channel`` coordinate to the current PyMC model context
        containing the sorted channel names from the DataFrame.

        Parameters
        ----------
        mmm : Model
            The parent MMM instance (must have an active PyMC model context).
        """
        self._mmm_dims = mmm.dims
        channels = self.rf_channels

        # Validate disjointness with standard media channels (mirrors Meridian's
        # InputData._validate_media_channels which raises ValueError on overlap).
        # Use getattr so this works when mmm does not have channel_columns (e.g.
        # non-MMM models that satisfy the Model protocol).
        mmm_channel_columns: list[str] = getattr(mmm, "channel_columns", [])
        overlap = set(channels) & set(mmm_channel_columns)
        if overlap:
            raise ValueError(
                "R&F channel names must be disjoint from MMM channel_columns. "
                f"Found overlap: {sorted(overlap)}. "
                "R&F data is an alternative representation to impressions/spend "
                "for those channels — including both would double-count their "
                "contribution to the model."
            )

        # Add rf_channel coordinate to the model
        pymc_model = pm.modelcontext(None)
        pymc_model.add_coord(RF_CHANNEL_COORD, channels)

        freq_np, reach_np = self._build_raw_arrays(self.df_frequency_reach.copy(), mmm)
        pmd.Data(
            f"{self.prefix}_frequency_raw",
            freq_np,
            dims=(self.date_column, *mmm.dims, RF_CHANNEL_COORD),
        )
        pmd.Data(
            f"{self.prefix}_reach_raw",
            reach_np,
            dims=(self.date_column, *mmm.dims, RF_CHANNEL_COORD),
        )

    def create_effect(self, mmm: Model):
        """Build the Meridian-style R&F pipeline and return the aggregate effect.

        Transformation steps:

        1. Saturate frequency: ``frequency_sat = saturation(frequency_raw)``
        2. Effective exposure: ``effective_exposure_raw = reach_raw × frequency_sat``
        3. Adstock: ``effective_exposure_adstocked = adstock(effective_exposure_raw)``
        4. Scale: ``channel_contribution = beta × effective_exposure_adstocked``
        5. Sum: ``total_effect = Σ_channel channel_contribution``

        Parameters
        ----------
        mmm : Model
            The parent MMM instance.

        Returns
        -------
        tensor
            Tensor with dims ``(date, *mmm.dims)`` to be added to mu.
        """
        model = mmm.model
        freq_raw = model[f"{self.prefix}_frequency_raw"]
        reach_raw = model[f"{self.prefix}_reach_raw"]

        # 1. Saturate frequency ONLY (reach is linear)
        frequency_sat = self.saturation.apply(x=freq_raw, core_dim=self.date_column)
        pmd.Deterministic(
            f"{self.prefix}_frequency_sat",
            frequency_sat,
        )

        # 2. Element-wise product with linear reach
        effective_exposure_raw = reach_raw * frequency_sat
        pmd.Deterministic(
            f"{self.prefix}_effective_exposure_raw",
            effective_exposure_raw,
        )

        # 3. Adstock over the exposure signal (core dim is date, not rf_channel)
        effective_exposure_adstocked = self.adstock.apply(
            x=effective_exposure_raw, core_dim=self.date_column
        )
        pmd.Deterministic(
            f"{self.prefix}_effective_exposure_adstocked",
            effective_exposure_adstocked,
        )

        # 4. Per-channel beta scaling — ensure beta_prior has rf_channel dim
        beta_prior = self.beta_prior.deepcopy()
        beta_prior.dims = (RF_CHANNEL_COORD,)
        beta = beta_prior.create_variable(
            f"{self.prefix}_beta",
            xdist=True,
        )
        channel_contribution = pmd.Deterministic(
            f"{self.prefix}_channel_contribution",
            beta * effective_exposure_adstocked,
        )

        # 5. Aggregate over rf_channel → shape (date, *mmm.dims)
        return pmd.Deterministic(
            f"{self.prefix}_total_effect",
            channel_contribution.sum(dim=RF_CHANNEL_COORD),
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Update reach & frequency data for prediction (e.g. future) dates.

        Zero-fills any dates in ``model.coords["date"]`` that are beyond the
        historical data in ``df_frequency_reach``.  Does not mutate the stored
        DataFrame; construct a new effect instance if historical data change.

        Parameters
        ----------
        mmm : Model
            The parent MMM instance.
        model : pm.Model
            The PyMC model whose ``pm.Data`` nodes will be updated.
        X : xr.Dataset
            The predictor dataset for the prediction window (used to read
            the new date coordinate).
        """
        raw_dates = model.coords.get(self.date_column)
        if raw_dates is None:
            raise ValueError(
                f"Model missing '{self.date_column}' coordinate during set_data."
            )
        new_dates = pd.to_datetime(list(raw_dates))
        df_extended = self.df_frequency_reach.copy()
        max_original_date = df_extended[self.date_column].max()

        future_mask = new_dates > max_original_date
        if future_mask.any():
            add_rows = []
            channels = self.rf_channels

            if self._mmm_dims is not None:
                dims_product = [list(model.coords[d]) for d in self._mmm_dims]  # type: ignore[arg-type]
            else:
                dims_product = []

            dim_combos = list(product(*dims_product)) if dims_product else [()]
            for date in new_dates[future_mask]:
                for combo in dim_combos:
                    for ch in channels:
                        row: dict[str, Any] = {
                            self.date_column: date,
                            self.channel_column: ch,
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

        freq_np, reach_np = self._build_raw_arrays(df_extended, mmm, model=model)
        # pmd.Data creates XTensorSharedVariable; pm.set_data doesn't handle it,
        # so update the shared variables directly via set_value.
        model[f"{self.prefix}_frequency_raw"].set_value(freq_np)
        model[f"{self.prefix}_reach_raw"].set_value(reach_np)

    # ------------------------------------------------------------------
    # Optimizer support
    # ------------------------------------------------------------------

    def build_rf_optimization_tensors(
        self,
        rf_budgets: Any,
        num_periods: int,
        budget_distribution: Any | None,
    ) -> tuple[Any, Any]:
        """Build (reach, frequency) PyTensor tensors for optimizer graph injection.

        Given a per-channel spend tensor (from the optimizer's free variables),
        converts spend to reach and frequency arrays that can replace the
        ``{prefix}_reach_raw`` and ``{prefix}_frequency_raw`` ``pm.Data``
        nodes via ``pm.do()``.

        Conversion chain
        ----------------
        .. code-block:: text

            spend → ÷ cost_per_unit → impressions
            impressions = reach × frequency*
            frequency* = assumed_frequency (fixed)
            reach = impressions / frequency*

        Parameters
        ----------
        rf_budgets : XTensorVariable
            Optimizer budget tensor for this effect, shape
            ``(n_rf_channels,)``.  Values are in monetary units.
        num_periods : int
            Number of optimisation time periods.
        budget_distribution : XTensorVariable or None
            Per-period distribution factors, shape ``(date, rf_channel)``.
            If ``None`` budgets are distributed uniformly.

        Returns
        -------
        reach_tensor : XTensorVariable
            Shape ``(date + l_max, *budget_dims_without_date, rf_channel)``.
        frequency_tensor : XTensorVariable
            Same shape as reach_tensor — constant assumed frequency tiled.

        Raises
        ------
        ValueError
            If ``cost_per_unit`` is not set on this effect.
        """
        if self.cost_per_unit is None:
            raise ValueError(
                f"FrequencyReachAdditiveEffect '{self.prefix}' requires "
                "'cost_per_unit' for budget optimization."
            )

        channels = self.rf_channels
        n_rf = len(channels)
        l_max = self.adstock.l_max

        # cost_per_unit: shape (n_rf,) — average over date dimension
        cpu_da = self.cost_per_unit_xarray  # (date, rf_channel)
        if cpu_da is None:
            raise ValueError(
                f"FrequencyReachAdditiveEffect '{self.prefix}' requires "
                "'cost_per_unit' to build optimization tensors."
            )
        cpu_mean = cpu_da.mean("date").values.astype("float64")  # (n_rf,)
        cpu_tensor = ptx.as_xtensor(
            pt.as_tensor_variable(cpu_mean, dtype="float64"),
            dims=(RF_CHANNEL_COORD,),
        )

        # assumed frequency: shape (n_rf,)
        freq_assumed = self.get_assumed_frequency_array()  # (n_rf,)
        freq_tensor = ptx.as_xtensor(
            pt.as_tensor_variable(freq_assumed, dtype="float64"),
            dims=(RF_CHANNEL_COORD,),
        )

        # impressions per period = budget / cost_per_unit
        impressions_per_period = rf_budgets / cpu_tensor  # (rf_channel,)

        # Expand to (date, rf_channel)
        if budget_distribution is not None:
            # budget_distribution has dims (date, rf_channel), sums to 1 along date
            impressions = budget_distribution * (
                impressions_per_period * num_periods
            )  # (date, rf_channel)
        else:
            impressions = impressions_per_period.expand_dims(
                date=num_periods
            )  # (date, rf_channel)

        # reach = impressions / assumed_frequency; frequency = constant
        # Guard against zero frequency (clip to small value)
        freq_safe_values = pt.clip(freq_tensor.values, 1e-6, np.inf)
        freq_safe_xt = ptx.as_xtensor(
            freq_safe_values,
            dims=(RF_CHANNEL_COORD,),
        )

        reach = impressions / freq_safe_xt  # (date, rf_channel)
        frequency = freq_safe_xt.expand_dims(date=num_periods)  # (date, rf_channel)

        # Append l_max carryover zeros
        zeros = ptx.as_xtensor(
            pt.zeros((l_max, n_rf), dtype="float32"), dims=("date", RF_CHANNEL_COORD)
        )
        reach_full = ptx.concat(
            [reach.astype("float32"), zeros], dim="date"
        )  # (date + l_max, rf_channel)
        frequency_full = ptx.concat(
            [frequency.astype("float32"), zeros], dim="date"
        )  # (date + l_max, rf_channel)

        return reach_full, frequency_full

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_raw_arrays(
        self, df: pd.DataFrame, mmm: Model, model: pm.Model | None = None
    ) -> tuple[Any, Any]:
        """Return reshaped (frequency, reach) arrays aligned to model coordinates.

        Used by both :meth:`create_data` and :meth:`set_data`.

        Parameters
        ----------
        df : pd.DataFrame
            Copy of ``df_frequency_reach`` (possibly extended with future rows).
        mmm : Model
            The parent MMM instance.

        Returns
        -------
        freq_np : np.ndarray
            Shape ``(n_dates, *dim_sizes, n_rf_channels)``, dtype float32.
        reach_np : np.ndarray
            Same shape, dtype float32.
        """
        model = model if model is not None else mmm.model
        channel_coord = self.rf_channels

        raw_dates = model.coords.get(self.date_column)
        if raw_dates is None:
            raise ValueError(f"Model missing '{self.date_column}' coordinate.")
        model_dates = pd.to_datetime(list(raw_dates))

        # Keep only allowed columns
        allowed_cols = {
            self.date_column,
            self.channel_column,
            "frequency",
            "reach",
            *mmm.dims,
        }
        drop_cols = [c for c in df.columns if c not in allowed_cols]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        group_keys = [self.date_column, *mmm.dims, self.channel_column]
        df = (
            df.groupby(group_keys, dropna=False)[["frequency", "reach"]]
            .mean()
            .reset_index()
        )

        # Build Cartesian product index over (date, *dims, rf_channel)
        iterables: list[list[Any]] = [list(model_dates)]
        for dim in mmm.dims:
            coord_vals = model.coords.get(dim)
            if coord_vals is None:
                raise ValueError(
                    f"Model missing dim coordinate '{dim}' required by MMM dims"
                )
            iterables.append(list(coord_vals))
        iterables.append(list(channel_coord))

        index_names = [self.date_column, *mmm.dims, self.channel_column]
        full_index = pd.MultiIndex.from_product(iterables, names=index_names)

        df = df.set_index(group_keys).reindex(full_index)
        df[["frequency", "reach"]] = df[["frequency", "reach"]].fillna(0.0)
        df = df.reset_index()

        df = df.set_index(index_names)
        freq_wide = (
            df["frequency"].unstack(self.channel_column).reindex(columns=channel_coord)
        )
        reach_wide = (
            df["reach"].unstack(self.channel_column).reindex(columns=channel_coord)
        )

        n_dates = len(model_dates)
        dim_sizes = []
        for d in mmm.dims:
            coord_vals = model.coords.get(d)
            if coord_vals is None:
                raise ValueError(
                    f"Model missing coordinate for dim '{d}' while reshaping tensor."
                )
            dim_sizes.append(len(list(coord_vals)))
        n_channels = len(channel_coord)

        expected_rows = n_dates * (int(np.prod(dim_sizes)) if dim_sizes else 1)
        if freq_wide.shape[0] != expected_rows or reach_wide.shape[0] != expected_rows:
            raise ValueError(
                "Internal shape mismatch while constructing reach/frequency tensors: "
                f"expected {expected_rows} rows, "
                f"found freq={freq_wide.shape[0]}, reach={reach_wide.shape[0]}."
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


__all__ = ["RF_CHANNEL_COORD", "FrequencyReachAdditiveEffect"]
