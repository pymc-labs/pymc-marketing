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
"""Campaign-granularity media effect with hierarchical pooling under channels.

``NestedCampaignMedia`` gives an MMM a short-term directional signal at
campaign level without destabilising channel-level ROI.  Campaign-level
identification comes from per-campaign saturation (the mix matters because
``sum(f(x_c)) != f(sum(x_c))``), extra model dims, and lift tests — combined
with hierarchical pooling of campaign parameters under channel hyperpriors.

Design notes
------------
- The effect owns its data variable (``campaign_data`` with dims
  ``("date", *mmm.dims, campaign_dim)``) and is meant to *replace* the
  built-in channel media term for the channels it covers.  Channel-level
  contributions are recovered as ``f"{prefix}_channel_contribution"``.
- Campaign spend is scaled by the *channel* total (max over dates), not per
  campaign, so priors mean the same thing for every campaign in a channel.
- Any :class:`~pymc_marketing.mmm.components.saturation.SaturationTransformation`
  can be used (Michaelis-Menten by default). Its priors live at *channel*
  level and are gathered to campaigns through the parent index; its own
  amplitude parameter is the channel capacity.
- The campaign curve is the channel curve scaled by campaign size on both
  axes::

      contribution_c(x) = cap_c**rho * mult_c * S(x / (cap_c**rho * scale_mult_c))

  where ``cap_c`` is the campaign's max channel-scaled spend and ``S`` the
  shared channel curve.  This degree-1 homogeneity makes the parameterisation
  *split-invariant* (splitting a campaign into parts with the same total
  spend leaves the channel contribution unchanged) and makes marginal
  returns equal across campaigns at proportional spend — for any saturation
  shape.  Deviations from that neutral point must be earned from data
  (flighting contrasts, covariates, lift tests), not from the
  parameterisation.
- Campaign multipliers are non-centred under channel pooling scales and, by
  default, constrained to a spend-share-weighted zero sum within each
  channel.
- ``incrementality_spec`` stays ``None``: ``channel_data`` is never an
  ancestor of this effect's contribution.  Point the budget optimizer at the
  effect via ``BudgetOptimizer(spend_vars=["campaign_data"],
  response_variable="total_response_original_scale")``; the default response
  variable does not depend on ``campaign_data`` and raises at construction.
- Adstock is not yet applied (identification of the campaign split rests on
  saturation, not carryover); pre-transform the data if carryover is needed.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pydantic import Field, InstanceOf
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import DataVarMuEffect, Model
from pymc_marketing.mmm.components.saturation import (
    MichaelisMentenSaturation,
    SaturationTransformation,
)
from pymc_marketing.mmm.distributions import DimWeightedZeroSumNormal
from pymc_marketing.mmm.lift_test import add_saturation_observations
from pymc_marketing.serialization import serialization

# Which function parameter plays the amplitude role, per saturation class.
# Used only to expose the f"{prefix}_beta_campaign" deterministic; shapes
# not listed here still work, they just skip that convenience deterministic.
_AMPLITUDE_PARAM: dict[str, str] = {
    "MichaelisMentenSaturation": "alpha",
    "LogisticSaturation": "beta",
    "InverseScaledLogisticSaturation": "beta",
    "TanhSaturationBaselined": "beta",
    "HillSaturation": "beta",
    "HillSaturationSigmoid": "beta",
    "RootSaturation": "beta",
    "LogSaturation": "beta",
    "NoSaturation": "beta",
}


def lognormal_relative_lift(
    name: str,
    mu: XTensorVariable,
    sigma: XTensorVariable,
    observed: XTensorVariable,
) -> XTensorVariable:
    """LogNormal lift likelihood with the noise as relative error on the measurement.

    The location is the log of the model's estimated lift ``mu``. The
    log-space scale is derived from the measurement's coefficient of
    variation ``sigma / observed``, a fixed number, so an estimated lift
    that collapses towards zero is penalised quadratically in log space. A
    LogNormal or Gamma moment-matched to ``mu`` and ``sigma`` instead lets
    the spread grow as ``mu`` shrinks and barely penalises that collapse,
    which gives the posterior a degenerate mode.

    The likelihood is median-matched: its median is ``mu`` and its mean is
    ``mu * exp(sigma_log**2 / 2)``, a ``+0.5%`` bias at a 10% coefficient of
    variation. Rows must satisfy the lift-table contract checked by the hook.
    """
    sigma_log = pmd.math.sqrt(pmd.math.log1p((sigma / observed) ** 2))
    return pmd.LogNormal(name, mu=pmd.math.log(mu), sigma=sigma_log, observed=observed)


def _check_lift_rows(df_lift_test: pd.DataFrame) -> None:
    """Enforce the lift-table contract: finite values, ``sigma > 0``, nonzero deltas.

    Rows outside this contract cannot be scored by a lift likelihood and
    would only surface as a ``-inf`` model logp at ``sample()`` time with
    nothing pointing at the offending row.
    """
    cols = ["x", "delta_x", "delta_y", "sigma"]
    values = df_lift_test[cols].to_numpy(dtype=float)
    bad = ~np.isfinite(values).all(axis=1)
    bad |= values[:, cols.index("sigma")] <= 0
    bad |= values[:, cols.index("delta_x")] == 0
    bad |= values[:, cols.index("delta_y")] == 0
    if bad.any():
        raise ValueError(
            "df_lift_test rows must have finite x, delta_x, delta_y and sigma, "
            "with sigma > 0 and nonzero delta_x and delta_y; offending rows: "
            f"{df_lift_test.index[bad].tolist()}"
        )


class NestedCampaignMedia(DataVarMuEffect):
    """Media effect at campaign granularity, hierarchically pooled by channel.

    Parameters
    ----------
    campaign_to_channel : dict[str, str]
        Maps each campaign name to its parent channel name.  Must cover
        exactly the campaigns present in the data variable's campaign
        coordinate.  Ragged channels (different campaign counts) are fine.
    saturation : SaturationTransformation, optional
        Any saturation from :mod:`pymc_marketing.mmm.components.saturation`;
        Michaelis-Menten by default.  Its priors are created at *channel*
        level (dims default to the effect's channel coordinate) and gathered
        to campaigns; its amplitude parameter is the channel capacity in
        scaled-spend, scaled-target units.  The saturation is evaluated on
        size-normalized spend ``x / cap_c**rho`` in ``[0, 1]``, so the
        library's default priors are sensible for every campaign.
    data_vars : list[str]
        Single data variable in ``mmm.xarray_dataset`` holding campaign
        spend with dims ``("date", *mmm.dims, campaign_dim)``.
    prefix : str
        Prefix for all model variable names created by this effect.
    campaign_dim, channel_dim : str
        Names of the campaign and channel dimensions.
    tau_beta_sigma, tau_lam_sigma : float
        Scales of the HalfNormal priors on the pooling strength of the
        campaign-level amplitude and x-scale multipliers.  Smaller means
        stronger pooling.  Under the spend-share-weighted zero sum the
        prior sd of a campaign's log-multiplier is ``tau * sqrt(1 - u_c**2)``
        with ``u = share / |share|``, so it is not uniform across campaigns:
        the dominant campaign of a channel is pinned close to the channel
        mean (for a 90/5/3/2 split about 12x tighter than its siblings).
        This protects the channel total; ``tau_beta_sigma`` is therefore not
        "the campaign multiplier scale" for every campaign.
    rho : float
        Exponent tying each campaign's curve to its size (``cap_c``, the
        campaign's max channel-scaled spend).  At ``rho=1`` the campaign
        curve is the channel curve scaled by campaign size on both axes:
        split-invariant, with equal marginal returns at proportional spend.
    zero_sum_multipliers : bool
        When True (default), the campaign log-multipliers are constrained to
        a *spend-share-weighted* zero sum within each channel. This is a
        structural guarantee, not just a conditioning device: the campaign
        layer then preserves the spend-weighted geometric channel mean
        exactly, so introducing campaigns cannot move the channel-level
        contribution the design exists to protect (unweighted centring
        shifts it by double-digit percentages at lopsided spend splits). It
        also decorrelates the channel-level parameters from the dominant
        campaign's deviation, which is worth 6-9x ESS/sec at 90/5/3/2-style
        splits. Channels with a single campaign get multiplier 1 (fully
        pooled) automatically; so do campaigns with zero historical spend.
    covariate_var : str, optional
        Name of a variable in ``mmm.xarray_dataset`` with dims
        ``(campaign_dim, covariate_dim)`` holding per-campaign covariates
        (e.g. log impressions, log clicks, CTR).  They enter the campaign
        amplitude multiplier and are spend-share-weighted-centred *within
        each channel* at build time, so they reallocate efficiency between
        a channel's campaigns without moving the channel-level total.
        Standardize covariates beforehand so ``gamma_sigma`` means the same
        thing for each of them.
    covariate_dim : str
        Name of the covariate dimension.  Default ``"covariate"``.
    gamma_sigma : float
        Scale of the Normal prior on the covariate coefficients.
    """

    campaign_to_channel: dict[str, str]
    saturation: InstanceOf[SaturationTransformation] = Field(
        default_factory=MichaelisMentenSaturation
    )
    data_vars: list[str] = ["campaign_data"]
    prefix: str = "campaign_media"
    campaign_dim: str = "campaign"
    channel_dim: str = "channel"
    tau_beta_sigma: float = 0.5
    tau_lam_sigma: float = 0.5
    rho: float = 1.0
    zero_sum_multipliers: bool = True
    covariate_var: str | None = None
    covariate_dim: str = "covariate"
    gamma_sigma: float = 0.5

    model_config = {"arbitrary_types_allowed": True}

    def create_data(self, mmm: Model) -> None:
        """Register campaign spend plus static index/scale data variables."""
        # The numpy reductions below are positional and assume the campaign
        # dim is last; the model graph itself is dim-name based
        da = mmm.xarray_dataset[self.data_vars[0]].transpose(..., self.campaign_dim)
        campaigns = [str(c) for c in da.coords[self.campaign_dim].values]

        missing = set(campaigns) - set(self.campaign_to_channel)
        extra = set(self.campaign_to_channel) - set(campaigns)
        if missing or extra:
            raise ValueError(
                "campaign_to_channel must cover exactly the campaigns in "
                f"{self.data_vars[0]!r}; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        channels = list(dict.fromkeys(self.campaign_to_channel[c] for c in campaigns))
        model = mmm.model

        overlap = set(channels) & set(map(str, model.coords.get(self.channel_dim, ())))
        if overlap:
            warnings.warn(
                f"Channels {sorted(overlap)} are both decomposed into campaigns by "
                f"this effect and present in the model's {self.channel_dim!r} "
                "coordinate. The effect REPLACES the channel-level media term; "
                "keeping the channel in channel_columns double-counts its spend. "
                "Exclude decomposed channels from channel_columns.",
                UserWarning,
                stacklevel=2,
            )

        channel_coord_name = f"{self.prefix}_{self.channel_dim}"
        if channel_coord_name not in model.coords:
            model.add_coord(channel_coord_name, channels)
        parent_idx = np.array(
            [channels.index(self.campaign_to_channel[c]) for c in campaigns]
        )

        # channel-level saturation priors, gathered to campaigns later
        self.saturation = self.saturation.with_default_prior_dims((channel_coord_name,))
        self.saturation.prefix = f"{self.prefix}_saturation"

        super().create_data(mmm)

        # channel scale: max over everything but channel of the channel total
        onehot = (parent_idx[:, None] == np.arange(len(channels))[None, :]).astype(
            da.dtype
        )
        channel_total = (da.values @ onehot).max(axis=tuple(range(da.ndim - 1)))
        scale = np.where(channel_total > 0, channel_total, 1.0)

        # campaign size on the scaled axis, for the size tie
        cap = (da.values / scale[parent_idx]).max(axis=tuple(range(da.ndim - 1)))
        dead = [c for c, k in zip(campaigns, cap, strict=True) if not k > 0]
        if dead:
            warnings.warn(
                f"Campaigns {dead} have no spend in the data. Their spend "
                "contributes nothing to the likelihood; their parameters are "
                "pinned to the pooled channel values (multiplier 1, cap "
                "fallback 1), so any curve read off them is prior-only.",
                UserWarning,
                stacklevel=2,
            )
        cap = np.where(cap > 0, cap, 1.0)

        pmd.Data(f"{self.prefix}_parent_idx", parent_idx, dims=(self.campaign_dim,))
        pmd.Data(
            f"{self.prefix}_parent_onehot",
            onehot,
            dims=(self.campaign_dim, channel_coord_name),
        )
        pmd.Data(f"{self.prefix}_channel_scale", scale, dims=(channel_coord_name,))
        pmd.Data(f"{self.prefix}_campaign_cap", cap, dims=(self.campaign_dim,))

        if self.zero_sum_multipliers:
            total = da.values.reshape(-1, da.shape[-1]).sum(axis=0)
            for g, channel in enumerate(channels):
                idx = np.flatnonzero(parent_idx == g)
                w = total[idx]
                # dead campaigns are left out of the constraint: their
                # multiplier is pinned at 1 instead of adding an unidentified
                # free direction. A channel needs two live campaigns to have
                # any free direction at all.
                live = idx[w > 0]
                if len(live) < 2:
                    continue
                sub_dim = self._channel_campaign_dim(channel)
                if sub_dim not in model.coords:
                    model.add_coord(sub_dim, [campaigns[i] for i in live])
                pmd.Data(
                    f"{self.prefix}_spend_share_{channel}",
                    total[live] / total[live].sum(),
                    dims=(sub_dim,),
                )
                scatter = np.zeros((len(live), len(campaigns)), dtype=da.dtype)
                scatter[np.arange(len(live)), live] = 1.0
                pmd.Data(
                    f"{self.prefix}_scatter_{channel}",
                    scatter,
                    dims=(sub_dim, self.campaign_dim),
                )

        if self.covariate_var is not None:
            cov_da = mmm.xarray_dataset[self.covariate_var]
            if set(cov_da.dims) != {self.campaign_dim, self.covariate_dim}:
                raise ValueError(
                    f"{self.covariate_var!r} must have dims exactly "
                    f"({self.campaign_dim!r}, {self.covariate_dim!r}); "
                    f"got {cov_da.dims}"
                )
            cov = cov_da.transpose(self.campaign_dim, self.covariate_dim).values
            # spend-share-weighted centring within each channel: the covariate
            # term reallocates efficiency between a channel's campaigns but
            # cannot move the channel total
            total_spend = da.values.reshape(-1, da.shape[-1]).sum(axis=0)
            channel_spend = (total_spend @ onehot)[parent_idx]
            # an all-zero channel has no share to centre by; its campaigns get
            # the raw covariate and the channel mean stays finite
            share = total_spend / np.where(channel_spend > 0, channel_spend, 1.0)
            weighted_mean = (share[:, None] * cov).T @ onehot  # (n_cov, n_channel)
            cov_centred = cov - weighted_mean.T[parent_idx]
            if self.covariate_dim not in model.coords:
                model.add_coord(
                    self.covariate_dim,
                    [str(c) for c in cov_da.coords[self.covariate_dim].values]
                    if self.covariate_dim in cov_da.coords
                    else np.arange(cov.shape[1]),
                )
            pmd.Data(
                f"{self.prefix}_covariates",
                cov_centred,
                dims=(self.campaign_dim, self.covariate_dim),
            )

    def _channel_campaign_dim(self, channel: str) -> str:
        """Coordinate name of the live campaigns of ``channel``."""
        return f"{self.prefix}_{channel}_campaign"

    def _zero_sum_multiplier(
        self, model: pm.Model, name: str, channels: list[str]
    ) -> XTensorVariable:
        """Standardised log-multiplier with a spend-share-weighted zero sum per channel.

        One :class:`~pymc_marketing.mmm.distributions.DimWeightedZeroSumNormal`
        per channel, over that channel's live campaigns, scattered back to the
        campaign dimension. Campaigns outside every block (dead campaigns and
        single-campaign channels) get zero, i.e. multiplier one.
        """
        p = self.prefix
        if not channels:
            zeros = pmd.zeros_like(model[f"{p}_campaign_cap"])
            return pmd.Deterministic(f"{p}_{name}", zeros)
        parts = []
        for channel in channels:
            sub_dim = self._channel_campaign_dim(channel)
            z = DimWeightedZeroSumNormal(
                f"{p}_{name}_{channel}",
                weights=model[f"{p}_spend_share_{channel}"],
                core_dims=sub_dim,
            )
            parts.append((z * model[f"{p}_scatter_{channel}"]).sum(dim=sub_dim))
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return pmd.Deterministic(f"{p}_{name}", total)

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Update ``campaign_data`` for a new prediction window.

        A wide DataFrame cannot carry the ``(date, campaign)`` spend, so the
        default MMM prediction route hands this effect a dataset without it.
        Silently reusing the training spend would be wrong, and a window of a
        different length would fail deep inside PyTensor. Instead the spend is
        set to zero over the new dates, with a warning: the campaigns then
        contribute nothing. To predict with campaign spend, pass an
        ``xr.Dataset`` ``X`` that carries the variable, as the budget
        optimizer's own dataset does.
        """
        var_name = self.data_vars[0]
        if var_name in X.data_vars:
            super().set_data(mmm, model, X)
            return
        current = model[var_name].get_value()
        dims = model.named_vars_to_dims[var_name]
        new_shape = tuple(
            X.sizes["date"] if dim == "date" else size
            for dim, size in zip(dims, current.shape, strict=True)
        )
        warnings.warn(
            f"{var_name!r} is not in the prediction data (a DataFrame cannot carry "
            "it), so campaign spend is set to zero over the new dates and the "
            "campaigns contribute nothing. Pass an xr.Dataset X with "
            f"{var_name!r} to predict with campaign spend.",
            UserWarning,
            stacklevel=2,
        )
        pm.set_data({var_name: np.zeros(new_shape, dtype=current.dtype)}, model=model)

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Build the nested campaign media contribution."""
        model = mmm.model
        p = self.prefix
        channel_coord_name = f"{p}_{self.channel_dim}"

        x = model[self.data_vars[0]]
        parent_idx = model[f"{p}_parent_idx"]
        onehot = model[f"{p}_parent_onehot"]
        scale = model[f"{p}_channel_scale"]
        cap = model[f"{p}_campaign_cap"]

        def gather(var):
            return var[{channel_coord_name: parent_idx}]

        x_scaled = x / gather(scale)

        tau_beta = pmd.HalfNormal(f"{p}_tau_beta", sigma=self.tau_beta_sigma)
        tau_lam = pmd.HalfNormal(f"{p}_tau_lam", sigma=self.tau_lam_sigma)
        zero_sum_channels = [
            channel
            for channel in model.coords[channel_coord_name]
            if f"{p}_spend_share_{channel}" in model.named_vars
        ]
        if self.zero_sum_multipliers:
            # Branch on the flag alone: with no channel holding two live
            # campaigns there are no free directions and every multiplier is
            # pinned at 1, rather than silently falling back to free
            # per-campaign multipliers.
            z_beta = self._zero_sum_multiplier(model, "z_beta", zero_sum_channels)
            z_lam = self._zero_sum_multiplier(model, "z_lam", zero_sum_channels)
        else:
            z_beta = pmd.Normal(f"{p}_z_beta", 0.0, 1.0, dims=(self.campaign_dim,))
            z_lam = pmd.Normal(f"{p}_z_lam", 0.0, 1.0, dims=(self.campaign_dim,))

        log_mult = tau_beta * z_beta
        if self.covariate_var is not None:
            cov = model[f"{p}_covariates"]
            gamma = pmd.Normal(
                f"{p}_gamma", 0.0, self.gamma_sigma, dims=(self.covariate_dim,)
            )
            log_mult = log_mult + (cov * gamma).sum(dim=self.covariate_dim)
        beta_multiplier = pmd.Deterministic(
            f"{p}_beta_multiplier", pmd.math.exp(log_mult)
        )
        lam_multiplier = pmd.Deterministic(
            f"{p}_lam_multiplier", pmd.math.exp(tau_lam * z_lam)
        )

        # channel-level saturation parameters, gathered to campaigns
        shape_params = {
            name: gather(var)
            for name, var in self.saturation._create_distributions().items()
        }

        # the campaign curve is the channel curve scaled by campaign size on
        # both axes: split-invariant for any saturation shape
        size = cap**self.rho
        x_rel = x_scaled / (size * lam_multiplier)
        curve = self.saturation.function(x_rel, dim="date", **shape_params)
        campaign_contribution = pmd.Deterministic(
            f"{p}_campaign_contribution", size * beta_multiplier * curve
        )

        amplitude = _AMPLITUDE_PARAM.get(type(self.saturation).__name__)
        if amplitude is not None:
            pmd.Deterministic(
                f"{p}_beta_campaign",
                gather(model[self.saturation.variable_mapping[amplitude]])
                * size
                * beta_multiplier,
            )
        if isinstance(self.saturation, MichaelisMentenSaturation):
            # half-saturation point of campaign c on the scaled-spend axis
            pmd.Deterministic(
                f"{p}_lam_campaign",
                gather(model[self.saturation.variable_mapping["lam"]])
                * size
                * lam_multiplier,
            )

        pmd.Deterministic(
            f"{p}_channel_contribution",
            (campaign_contribution * onehot).sum(dim=self.campaign_dim),
        )
        return pmd.Deterministic(
            f"{p}_effect_contribution",
            campaign_contribution.sum(dim=self.campaign_dim),
        )

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        mmm: Model,
        dist: Callable[..., XTensorVariable] = lognormal_relative_lift,
        name: str | None = None,
        target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "NestedCampaignMedia":
        """Calibrate campaign saturation curves with lift-test results.

        For each row, the model's estimated lift on the campaign's own curve,
        ``cap**rho * mult * (S(x + delta_x) - S(x))``, is conditioned on the
        measured ``delta_y`` with ``dist``. Works for any saturation shape.
        This is the strongest campaign-level identification source: it does
        not rely on flighting contrasts in the historical spend.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            One row per lift test with columns ``{campaign_dim}``, ``x``,
            ``delta_x``, ``delta_y``, ``sigma``. ``x`` and ``delta_x`` are in
            spend units; they are scaled internally by the campaign's channel
            scale. ``delta_y`` and ``sigma`` are in target units.
        mmm : Model
            The MMM the effect was built into. The model must be built.
        dist : callable, optional
            Likelihood for the lift measurements, called with ``name``,
            ``mu`` (the estimated lift), ``sigma`` and ``observed``. By
            default :func:`lognormal_relative_lift`, a positive likelihood
            with the noise as relative error on the measurement. A
            ``pmd.Gamma`` or a moment-matched LogNormal is not recommended:
            they barely penalise an estimated lift near zero, which gives
            the posterior a degenerate mode with a collapsed half-saturation.
        name : str, optional
            Name of the likelihood, defaults to
            ``f"{prefix}_lift_measurements"``.
        target_transform : Callable, optional
            Function ``(n, 1) -> (n, 1)`` scaling ``delta_y``/``sigma`` into
            the model's (scaled) target units. Defaults to dividing by
            ``mmm.scalers._target`` when available (per-dim aware; model dim
            columns are then required in ``df_lift_test``), otherwise the
            identity (targets assumed unscaled).
        """
        model = mmm.model
        p = self.prefix
        channel_coord_name = f"{p}_{self.channel_dim}"
        if f"{p}_beta_multiplier" not in model.named_vars:
            raise RuntimeError(
                "The model has not been built yet. Build the model before "
                "adding lift test measurements."
            )
        required = {self.campaign_dim, "x", "delta_x", "delta_y", "sigma"}
        missing = required - set(df_lift_test.columns)
        if missing:
            raise KeyError(f"df_lift_test is missing columns {sorted(missing)}")

        campaigns = [str(c) for c in model.coords[self.campaign_dim]]
        unknown = set(df_lift_test[self.campaign_dim].astype(str)) - set(campaigns)
        if unknown:
            raise ValueError(f"Unknown campaigns in df_lift_test: {sorted(unknown)}")
        _check_lift_rows(df_lift_test)

        scale = np.asarray(model[f"{p}_channel_scale"].get_value())
        parent = np.asarray(model[f"{p}_parent_idx"].get_value()).astype(int)
        scale_map = dict(zip(campaigns, scale[parent], strict=True))
        row_scale = (
            df_lift_test[self.campaign_dim].astype(str).map(scale_map).to_numpy()
        )

        if target_transform is None:
            scalers = getattr(mmm, "scalers", None)
            if scalers is not None and hasattr(scalers, "_target"):
                target_scale = scalers._target
                dims = tuple(getattr(mmm, "dims", ()) or ())
                if dims:
                    missing_dims = set(dims) - set(df_lift_test.columns)
                    if missing_dims:
                        raise KeyError(
                            f"df_lift_test is missing the model dim columns "
                            f"{sorted(missing_dims)} needed to scale delta_y/sigma"
                        )
                    row_target_scale = (
                        target_scale.sel(
                            {
                                d: xr.DataArray(
                                    df_lift_test[d].to_numpy(), dims="__row__"
                                )
                                for d in dims
                            }
                        )
                        .to_numpy()
                        .reshape(-1, 1)
                    )
                else:
                    row_target_scale = float(target_scale)

                def target_transform(values: np.ndarray) -> np.ndarray:
                    return values / row_target_scale
            else:

                def target_transform(values: np.ndarray) -> np.ndarray:
                    return values

        def scale_target(col: pd.Series) -> np.ndarray:
            return target_transform(col.to_numpy().reshape(-1, 1)).flatten()

        df_scaled = df_lift_test.assign(
            **{
                "x": df_lift_test["x"] / row_scale,
                "delta_x": df_lift_test["delta_x"] / row_scale,
                "delta_y": scale_target(df_lift_test["delta_y"]),
                "sigma": scale_target(df_lift_test["sigma"]),
                # the saturation's channel-level params are indexed by the
                # effect's channel coordinate
                channel_coord_name: df_lift_test[self.campaign_dim]
                .astype(str)
                .map(self.campaign_to_channel),
            }
        )

        rho = self.rho
        saturation_function = self.saturation.function

        def campaign_curve(x, cap, beta_mult, lam_mult, **shape_params):
            size = cap**rho
            return (
                size
                * beta_mult
                * saturation_function(x / (size * lam_mult), **shape_params)
            )

        variable_mapping = {
            "cap": f"{p}_campaign_cap",
            "beta_mult": f"{p}_beta_multiplier",
            "lam_mult": f"{p}_lam_multiplier",
            **self.saturation.variable_mapping,
        }

        add_saturation_observations(
            df_scaled,
            variable_mapping=variable_mapping,
            saturation_function=campaign_curve,
            model=model,
            dist=dist,
            name=name or f"{p}_lift_measurements",
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        data = self.model_dump(mode="json", exclude={"saturation"})
        data["saturation"] = self.saturation.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NestedCampaignMedia":
        """Reconstruct from a dict."""
        work = {k: v for k, v in data.items() if k != "__type__"}
        if isinstance(work.get("saturation"), dict):
            work["saturation"] = serialization.deserialize(work["saturation"])
        return cls(**work)
