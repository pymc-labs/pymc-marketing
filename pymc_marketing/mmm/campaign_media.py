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
  effect via ``BudgetOptimizer(spend_vars=["campaign_data"])``.
- Adstock is not yet applied (identification of the campaign split rests on
  saturation, not carryover); pre-transform the data if carryover is needed.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pymc.dims as pmd
import xarray as xr
from pydantic import Field, InstanceOf
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import DataVarMuEffect, Model
from pymc_marketing.mmm.components.saturation import (
    MichaelisMentenSaturation,
    SaturationTransformation,
)
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


def _weighted_zero_sum_basis(weights: np.ndarray) -> np.ndarray:
    """Orthonormal basis (k, k-1) of the hyperplane ``sum(weights * x) = 0``.

    Restriction of the Householder reflection sending ``u = w/|w|`` to
    ``-e_k``; the ``v = u + e_k`` sign choice keeps the denominator
    ``1 + u_k >= 1`` for non-negative weights.
    """
    w = np.asarray(weights, dtype=float)
    k = w.shape[0]
    if k < 2:
        return np.zeros((k, 0))
    u = w / np.linalg.norm(w)
    z0 = np.concatenate([np.eye(k - 1), np.zeros((1, k - 1))], axis=0)
    v = u.copy()
    v[-1] += 1.0
    coef = (v @ z0) / (1.0 + u[-1])
    return z0 - v[:, None] * coef[None, :]


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
        stronger pooling.
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
        da = mmm.xarray_dataset[self.data_vars[0]]
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
            blocks = []
            for g in range(len(channels)):
                idx = np.flatnonzero(parent_idx == g)
                w = total[idx]
                live = np.flatnonzero(w > 0)
                # dead campaigns keep zero basis rows: their multiplier is
                # pinned at 1 instead of adding an unidentified free direction
                blocks.append((idx[live], _weighted_zero_sum_basis(w[live])))
            n_free = sum(b.shape[1] for _, b in blocks)
            free_dim = f"{self.prefix}_free"
            if n_free > 0:
                basis = np.zeros((len(campaigns), n_free))
                pos = 0
                for idx, b in blocks:
                    basis[idx, pos : pos + b.shape[1]] = b
                    pos += b.shape[1]
                if free_dim not in model.coords:
                    model.add_coord(free_dim, np.arange(n_free))
                pmd.Data(
                    f"{self.prefix}_mult_basis",
                    basis,
                    dims=(self.campaign_dim, free_dim),
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
            share = total_spend / (total_spend @ onehot)[parent_idx]
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
        basis_name = f"{p}_mult_basis"
        if self.zero_sum_multipliers and basis_name in model.named_vars:
            basis = model[basis_name]
            free_dim = f"{p}_free"
            z_beta_free = pmd.Normal(f"{p}_z_beta", 0.0, 1.0, dims=(free_dim,))
            z_lam_free = pmd.Normal(f"{p}_z_lam", 0.0, 1.0, dims=(free_dim,))
            z_beta = (basis * z_beta_free).sum(dim=free_dim)
            z_lam = (basis * z_lam_free).sum(dim=free_dim)
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
        dist: type[pmd.DimDistribution] = pmd.Gamma,
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
        dist : pymc.dims.DimDistribution class, optional
            Likelihood for the lift measurements, by default ``pmd.Gamma``.
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
