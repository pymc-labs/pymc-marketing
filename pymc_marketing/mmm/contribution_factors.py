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
"""Historical contribution factors: specification, graph and persistence boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import warnings
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pymc.model.fgraph import clone_model
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc_extras.prior import Prior
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm.data_conversion import to_mmm_dataset
from pymc_marketing.mmm.spend_reach import resolve_channel_dependent_effects
from pymc_marketing.serialization import serialization

if TYPE_CHECKING:
    from pymc_marketing.mmm.mmm import MMM


def _validate_factor_spec(
    name: str,
    channels: Collection[str],
    quantity: str,
    prior: Prior,
    *,
    channel_columns: Collection[str],
    reserved_names: Collection[str],
) -> dict[str, Any]:
    """Normalize a scalar specification without changing a model graph."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("name must be an identifier starting with a letter")
    if name.endswith(("_original_scale", "_factor")) or {
        name,
        f"{name}_factor",
    }.intersection(reserved_names):
        raise ValueError(
            f"Contribution factor name {name!r} is already in use or reserved"
        )
    if (
        isinstance(channels, str)
        or not isinstance(channels, Collection)
        or len(channels) == 0
    ):
        raise ValueError("channels must be a nonempty one-dimensional collection")
    channels = list(channels)
    if any(not isinstance(c, str) for c in channels) or len(set(channels)) != len(
        channels
    ):
        raise ValueError("channels must contain distinct channel names")
    if set(channels).difference(channel_columns):
        raise ValueError("channels contains unknown channel names")
    if quantity not in ("total", "share"):
        raise ValueError("quantity must be 'total' or 'share'")
    if (
        not isinstance(prior, Prior)
        or prior.dims
        or prior.transform
        or not prior.centered
    ):
        raise ValueError("prior must be a centered scalar Prior without a transform")
    if any(
        not isinstance(v, (int, float, np.number)) or not np.isfinite(v)
        for v in prior.parameters.values()
    ):
        raise ValueError("prior parameters must be fixed finite scalars")
    if prior.preliz.support != (-np.inf, np.inf):
        raise ValueError(
            "Only smooth real-line components are supported; bounded components "
            "(including Beta) and hard caps are unsupported"
        )
    distribution = prior.pymc_distribution.dist(**prior.parameters)
    if distribution.ndim or not distribution.dtype.startswith("float"):
        raise ValueError("prior must be a scalar continuous distribution")
    pm.logp(distribution, 0.0).eval()
    return {
        "name": name,
        "channels": channels,
        "quantity": quantity,
        "prior": prior.to_dict(),
    }


def _training_fingerprint(dataset: xr.Dataset, mask: xr.DataArray) -> str:
    """Hash named cells canonically, independently of physical axis order."""
    digest = hashlib.sha256()
    for name, value in sorted(dataset.data_vars.items()):
        value = value.transpose(*sorted(value.dims))
        digest.update(json.dumps([name, list(value.dims)]).encode())
        digest.update(
            np.ascontiguousarray(value.fillna(0).values, dtype="<f8").tobytes()
        )
    for name, coord in sorted(dataset.coords.items()):
        values = (
            [pd.Timestamp(v).isoformat() for v in coord.values]
            if name == "date"
            else coord.values.tolist()
        )
        digest.update(json.dumps([name, values], default=str).encode())
    digest.update(
        np.ascontiguousarray(mask.transpose(*sorted(mask.dims)), dtype=bool).tobytes()
    )
    return digest.hexdigest()


@dataclass
class ContributionFactorState:
    """Own component definitions and the fixed reference, never a fitted posterior."""

    specs: list[dict[str, Any]] = field(default_factory=list)
    mask: xr.DataArray | None = None
    fingerprint: str | None = None
    inputs: dict[str, np.ndarray] = field(default_factory=dict)
    frozen_graph: Any = None
    restoring: bool = False

    def validate_reference(self, mmm: MMM, X: Any, y: Any) -> None:
        """Refuse changed-period refits/CV before modifying any model state."""
        if not self.specs:
            return
        candidate = to_mmm_dataset(
            X,
            y,
            date_column=mmm.date_column,
            dims=mmm.dims,
            channel_columns=mmm.channel_columns,
            control_columns=mmm.control_columns,
            target_column=mmm.target_column,
        )
        mask = self.mask if self.restoring else np.isfinite(candidate["_target"])
        if mask is None or _training_fingerprint(candidate, mask) != self.fingerprint:
            raise ValueError(
                "Contribution factors refer to the original training cells. "
                "Refitting or time-slice cross-validation on different data requires "
                "a fresh MMM and period-specific factors."
            )

    @contextmanager
    def rebuilding(self, mmm: MMM, X: Any, y: Any) -> Iterator[None]:
        """Build into fresh state; publish it only after every factor restores."""
        self.validate_reference(mmm, X, y)
        with self.transaction(mmm):
            candidate = ContributionFactorState()
            mmm._contribution_factors = candidate
            yield
            if self.restoring:
                candidate.mask = self.mask
            candidate._capture_inputs(mmm)
            if self.specs:
                candidate._validate_model(mmm)
                for spec in self.specs:
                    candidate._register_factor(mmm, spec)

    @staticmethod
    @contextmanager
    def transaction(mmm: MMM) -> Iterator[None]:
        """Restore the previous model and definitions after an unsuccessful update."""
        previous = mmm.__dict__.copy()
        try:
            yield
        except Exception:
            mmm.__dict__.clear()
            mmm.__dict__.update(previous)
            raise

    def _capture_inputs(self, mmm: MMM) -> None:
        """Copy only mutable ancestors of the original-scale contribution."""
        source = mmm.model["channel_contribution"] * mmm.model["target_scale"]
        inputs = set(ancestors([source]))
        self.inputs = {
            v.name: np.array(v.get_value(), copy=True)
            for v in mmm.model.data_vars
            if v in inputs
        }

    def _validate_model(self, mmm: MMM) -> None:
        """Check the built attribution graph, allowing spend-independent effects."""
        if mmm.link != "identity":
            raise ValueError(
                "Contribution factors require additive identity-link models"
            )
        if resolve_channel_dependent_effects(mmm):
            raise ValueError(
                "Contribution factors do not support spend-dependent MuEffects"
            )
        # The resolver deliberately skips unlocatable custom effects. A factor
        # cannot promise direct attribution for an effect it cannot inspect.
        for effect in mmm.mu_effects:
            try:
                name = effect.contribution_var_name
            except (AttributeError, NotImplementedError) as err:
                raise ValueError(
                    "Contribution factors require inspectable MuEffects"
                ) from err
            if name not in mmm.model.named_vars:
                raise ValueError(
                    "Contribution factors require inspectable MuEffects; "
                    f"contribution node {name!r} was not found"
                )

    def add(
        self,
        mmm: MMM,
        name: str,
        channels: Collection[str],
        quantity: str,
        prior: Prior,
    ) -> None:
        """Validate everything before attaching nodes or committing definitions."""
        if mmm.idata is not None and "posterior" in mmm.idata:
            raise ValueError(
                "Add contribution factors before fitting; the existing posterior would be stale"
            )
        self._validate_model(mmm)
        spec = _validate_factor_spec(
            name,
            channels,
            quantity,
            prior,
            channel_columns=mmm.channel_columns,
            reserved_names=set(mmm.model.named_vars) | set(mmm.model.coords),
        )
        if not np.all(np.isfinite(self.inputs["target_scale"])):
            raise ValueError(
                "Original-unit contributions require finite target scaling"
            )
        if self.mask is None or not bool(self.mask.any()):
            raise ValueError("Contributions require finite observed training outcomes")
        denominator = float(mmm.xarray_dataset["_target"].where(self.mask, 0).sum())
        if quantity == "share" and (not np.isfinite(denominator) or denominator <= 0):
            raise ValueError(
                "Shares require a finite positive sum of observed training outcomes"
            )
        # Only make a sign claim when this model's amplitude support establishes it.
        beta = mmm.model_config.get("saturation_beta")
        if isinstance(beta, Prior) and beta.preliz.support[0] >= 0:
            if (
                float(
                    -np.expm1(
                        pm.logcdf(
                            prior.pymc_distribution.dist(**prior.parameters), 0
                        ).eval()
                    )
                )
                < 0.001
            ):
                warnings.warn(
                    "This component assigns less than 0.1% probability to nonnegative "
                    "contributions; check the belief and its units.",
                    UserWarning,
                    stacklevel=3,
                )
        self._register_factor(mmm, spec)

    def _freeze_training_contribution_graph(self, mmm: MMM) -> Any:
        """Reuse the original-scale transform, sharing RVs but fixing Data."""
        source = mmm._link_spec.original_scale_transform(
            mmm.model["channel_contribution"], mmm.model["target_scale"]
        )
        replacements = {
            mmm.model[k]: mmm.model[k].type.constant_type(
                type=mmm.model[k].type, data=v
            )
            for k, v in self.inputs.items()
        }
        replacements.update({v: v for v in mmm.model.free_RVs})
        return graph_replace([source], replace=replacements, strict=False)[0]

    def _register_factor(self, mmm: MMM, spec: dict[str, Any]) -> None:
        """Register one scalar aggregate and differentiable log-density factor."""
        if self.frozen_graph is None:
            self.frozen_graph = self._freeze_training_contribution_graph(mmm)
            self.fingerprint = _training_fingerprint(mmm.xarray_dataset, self.mask)
        indices = [mmm.channel_columns.index(c) for c in spec["channels"]]
        total = (
            self.frozen_graph.isel(channel=indices)
            .where(as_xtensor(self.mask), 0)
            .sum()
        )
        denominator = float(mmm.xarray_dataset["_target"].where(self.mask, 0).sum())
        value = total / denominator if spec["quantity"] == "share" else total
        prior = serialization.deserialize_model_config({"prior": spec["prior"]})[
            "prior"
        ]
        with mmm.model:
            aggregate = pmd.Deterministic(spec["name"], value)
            value_aggregate = mmm.model.replace_rvs_by_values([aggregate])[0]
            # XTensor Potential logp is not lowered by conditional_logp. Keep
            # this explicit lowering until upstream supports its pullbacks.
            # https://github.com/pymc-devs/pymc/issues/8506
            logp = rewrite_graph(
                pm.logp(
                    prior.pymc_distribution.dist(**prior.parameters),
                    value_aggregate.values,
                ),
                include=("lower_xtensor",),
            )
            pm.Potential(f"{spec['name']}_factor", logp)
        self.specs.append(copy.deepcopy(spec))

    def to_attrs(self) -> dict[str, str]:
        """Store small definitions and a fingerprint, with no numeric training arrays."""
        return {
            "contribution_factors": json.dumps(
                {"version": 1, "specs": self.specs, "fingerprint": self.fingerprint}
            )
        }

    def attach_mask(self, idata: xr.DataTree) -> None:
        """Store missingness as a labeled array, independently of zero-filled y."""
        if self.specs:
            idata["contribution_factor_data"] = xr.Dataset({"observed_mask": self.mask})

    @classmethod
    def from_idata(cls, idata: xr.DataTree) -> ContributionFactorState:
        """Read factor definitions; frozen graph inputs are rebuilt from fit_data."""
        metadata = json.loads(idata.attrs.get("contribution_factors", "{}"))
        if not metadata and json.loads(idata.attrs.get("contribution_priors", "[]")):
            raise ValueError(
                "Legacy contribution-prior metadata requires rebuilding with add_contribution_factor"
            )
        specs = metadata.get("specs", [])
        if metadata and metadata.get("version") != 1:
            raise ValueError("Unsupported contribution-factor serialization version")
        mask = idata.contribution_factor_data["observed_mask"].copy() if specs else None
        return cls(
            specs=specs,
            mask=mask,
            fingerprint=metadata.get("fingerprint"),
            restoring=True,
        )

    def sample(self, mmm: MMM, **kwargs: Any) -> xr.DataTree:
        """Construct an isolated historical sampling model, without observations."""
        if "model" in kwargs or kwargs.get("return_inferencedata") is False:
            raise ValueError(
                "sample_effective_prior controls its model and requires inference-data output"
            )
        expected = {f"{s['name']}_factor" for s in self.specs}
        if {v.name for v in mmm.model.potentials} != expected or len(
            mmm.model.observed_RVs
        ) != 1:
            raise ValueError(
                "Effective-prior sampling requires only contribution factors "
                "and the sales likelihood; additional likelihoods/Potentials are unsupported"
            )
        # Sampling needs a fresh graph/configuration, not a copy of a potentially
        # large fitted trace or the graph that is about to be rebuilt.
        state = ContributionFactorState(
            specs=copy.deepcopy(self.specs),
            mask=self.mask.copy() if self.mask is not None else None,
            fingerprint=self.fingerprint,
            restoring=True,
        )
        memo: dict[int, Any] = {id(self): state, id(mmm.model): None}
        if mmm.idata is not None:
            memo[id(mmm.idata)] = None
        working = copy.deepcopy(mmm, memo)
        working.idata = None
        working.build_model(working.xarray_dataset)
        # Preserve any user-added parameter RVs and Deterministics. Rebuilding
        # supplies historical Data values, but is not the sampling graph.
        sampling_copy = clone_model(mmm.model)
        with sampling_copy:
            pm.set_data(
                {v.name: v.get_value() for v in working.model.data_vars},
                coords=working.model.coords,
            )
        frozen = freeze_dims_and_data(sampling_copy)
        output = frozen[mmm.output_var]
        fixed = output.type.constant_type(
            type=output.type,
            data=np.zeros(
                tuple(len(frozen.coords[d]) for d in output.dims), dtype=output.dtype
            ),
        )
        sampling_model = pm.do(
            frozen, {mmm.output_var: fixed}, make_interventions_shared=False
        )
        if sampling_model.observed_RVs:
            raise ValueError(
                "Effective-prior sampling must not retain observed likelihoods"
            )
        with sampling_model:
            result = pm.sample(**kwargs)
        result["prior"] = result.posterior.dataset
        result = result.drop_nodes("posterior")
        if "sample_stats" in result:
            result["sample_stats_prior"] = result.sample_stats.dataset
            result = result.drop_nodes("sample_stats")
        result.attrs["sampling_target"] = (
            "parameter priors multiplied by historical contribution factors"
        )
        return result
