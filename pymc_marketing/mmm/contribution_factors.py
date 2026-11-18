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

from pymc_marketing.mmm._contribution_support import (
    ContributionSupport,
    is_identically_zero,
)
from pymc_marketing.mmm.data_conversion import to_mmm_dataset
from pymc_marketing.mmm.hsgp import HSGPBase
from pymc_marketing.mmm.spend_reach import resolve_channel_dependent_effects
from pymc_marketing.names.mmm import ORIGINAL_SCALE_SUFFIX, Attr, DatasetVar, Dim, Var
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
    if name.endswith((ORIGINAL_SCALE_SUFFIX, "_factor")) or {
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
    # Initially support only these two nonnegative families. This is an
    # explicit API allowlist, not a classification of all bounded densities.
    if prior.preliz.support != (-np.inf, np.inf) and prior.distribution not in (
        "LogNormal",
        "HalfNormal",
    ):
        raise ValueError(
            "Supported components have full real-line support or are LogNormal/HalfNormal. "
            "Finite-support components (including Beta) and hard caps are unsupported"
        )
    distribution = prior.pymc_distribution.dist(**prior.parameters)
    if distribution.ndim or not distribution.dtype.startswith("float"):
        raise ValueError("prior must be a scalar continuous distribution")
    pm.logp(distribution, 1.0).eval()
    return {
        "name": name,
        "channels": channels,
        "quantity": quantity,
        "prior": prior.to_dict(),
    }


def _training_fingerprint(dataset: xr.Dataset, mask: xr.DataArray) -> str:
    """Hash named cells canonically, independently of physical axis order."""
    dataset = dataset.sortby(list(dataset.dims))
    mask = mask.sortby(list(mask.dims))
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
            if name == Dim.DATE
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
    support: ContributionSupport | None = None
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
            extra_vars=mmm._effect_extra_vars(X),
            target_column=mmm.target_column,
        )
        mask = (
            self.mask if self.restoring else np.isfinite(candidate[DatasetVar.TARGET])
        )
        if mask is None or _training_fingerprint(candidate, mask) != self.fingerprint:
            raise ValueError(
                "Contribution factors refer to the original training cells. "
                "Refitting or time-slice cross-validation on different data requires "
                "a fresh MMM and period-specific factors."
            )

    @contextmanager
    def rebuilding(self, mmm: MMM, X: Any, y: Any) -> Iterator[MMM]:
        """Build into fresh state; publish it only after every factor restores."""
        self.validate_reference(mmm, X, y)
        # Stage library-owned graph/configuration fields. Effect hooks retain
        # normal MMM semantics and access the original caller-owned components.
        working = self.isolated_configuration(mmm) if self.specs else mmm
        candidate = ContributionFactorState()
        working._contribution_factors = candidate
        yield working
        if self.restoring:
            candidate.mask = self.mask
        candidate._capture_inputs(working)
        if self.specs:
            candidate._validate_model(working)
            for spec in self.specs:
                candidate._register_factor(working, spec)
        if working is not mmm:
            self.commit_configuration(mmm, working)

    @staticmethod
    def isolated_configuration(mmm: MMM) -> MMM:
        """Stage MMM-owned fields without deep-copying custom factories or resources."""
        working = copy.copy(mmm)
        working.model_config = mmm.model_config.copy()
        working.mu_effects = mmm.mu_effects.copy()
        # register_data replaces X on these factories; isolate that build-time
        # registration, while preserving the public factory identity on commit.
        for attribute in (Attr.TIME_VARYING_INTERCEPT, Attr.TIME_VARYING_MEDIA):
            factory = getattr(mmm, attribute)
            if isinstance(factory, HSGPBase):
                setattr(working, attribute, factory.model_copy())
        return working

    @staticmethod
    def commit_configuration(mmm: MMM, working: MMM) -> None:
        """Publish a staged graph without swapping caller-owned configuration."""
        config = working.model_config.copy()
        mmm.model_config.clear()
        mmm.model_config.update(config)
        working.model_config = mmm.model_config
        mmm.mu_effects[:] = working.mu_effects
        working.mu_effects = mmm.mu_effects
        for attribute in (Attr.TIME_VARYING_INTERCEPT, Attr.TIME_VARYING_MEDIA):
            original, staged = getattr(mmm, attribute), getattr(working, attribute)
            if (
                isinstance(original, HSGPBase)
                and type(staged) is type(original)
                and staged is not original
            ):
                original.__dict__.update(staged.__dict__)
                setattr(working, attribute, original)
        # Graph construction may remove cache fields; publish deletions too.
        mmm.__dict__.clear()
        mmm.__dict__.update(working.__dict__)

    @contextmanager
    def predicting(self, mmm: MMM) -> Iterator[None]:
        """Hide only PyMC's redundant warning for classified contribution factors."""
        expected = {f"{spec['name']}_factor" for spec in self.specs}
        only_factors = (
            bool(expected) and {v.name for v in mmm.model.potentials} == expected
        )
        with warnings.catch_warnings():
            if only_factors:
                warnings.filterwarnings(
                    "ignore",
                    message="The effect of Potentials on other parameters is ignored during .*predictive sampling.*",
                    category=UserWarning,
                )
            yield

    def _capture_inputs(self, mmm: MMM) -> None:
        """Copy only mutable ancestors of the original-scale contribution."""
        source = mmm.model[Var.CHANNEL_CONTRIBUTION] * mmm.model[Var.TARGET_SCALE]
        inputs = set(ancestors([source]))
        self.inputs = {
            v.name: np.array(v.get_value(), copy=True)
            for v in mmm.model.data_vars
            if v in inputs
        }

        self.support = ContributionSupport.capture(mmm)

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
            reserved_names=set(mmm.model.named_vars)
            | set(mmm.model.coords)
            | {"chain", "draw", "sample"},
        )
        if getattr(mmm, "_target_is_placeholder", False):
            raise ValueError(
                "Contributions require real observed outcomes, not a placeholder target"
            )
        if not np.all(np.isfinite(self.inputs[Var.TARGET_SCALE])) or not np.all(
            self.inputs[Var.TARGET_SCALE] > 0
        ):
            raise ValueError(
                "Original-unit contributions require finite positive target scaling"
            )
        if self.mask is None or not bool(self.mask.any()):
            raise ValueError("Contributions require finite observed training outcomes")
        denominator = float(
            mmm.xarray_dataset[DatasetVar.TARGET].where(self.mask, 0).sum()
        )
        if quantity == "share" and (not np.isfinite(denominator) or denominator <= 0):
            raise ValueError(
                "Shares require a finite positive sum of observed training outcomes"
            )
        # Only make a sign claim when this model's amplitude support establishes it.
        if (
            self.support is not None
            and self.support.nonnegative_error(list(channels)) is None
        ):
            try:
                log_negative_mass = pm.logcdf(
                    prior.pymc_distribution.dist(**prior.parameters), 0
                ).eval()
            except NotImplementedError:
                # A CDF is optional for this advisory diagnostic. Some supported
                # smooth densities (e.g. SkewNormal) only implement logp in PyMC.
                pass
            else:
                if float(-np.expm1(log_negative_mass)) < 0.001:
                    warnings.warn(
                        "This component assigns less than 0.1% probability to nonnegative "
                        "contributions; check the belief and its units.",
                        UserWarning,
                        stacklevel=3,
                    )
        self._register_factor(mmm, spec)

    def _validate_component_support(
        self, mmm: MMM, channels: Collection[str], prior: Prior
    ) -> None:
        """Prevent nonnegative factors from truncating signed channel effects."""
        if prior.distribution in ("LogNormal", "HalfNormal"):
            if self.support is None:
                raise RuntimeError("Historical contribution support was not captured")
            reason = self.support.nonnegative_error(list(channels))
            if reason is not None:
                raise ValueError(
                    f"LogNormal/HalfNormal factors require nonnegative contributions: {reason}"
                )
            if prior.distribution == "LogNormal":
                if self.mask is None or not self.support.has_positive_target(
                    list(channels), self.mask
                ):
                    raise ValueError(
                        "LogNormal factors require nonzero historical channel inputs reaching observed outcomes"
                    )

    def _freeze_training_contribution_graph(self, mmm: MMM) -> Any:
        """Reuse the original-scale transform, sharing RVs but fixing Data."""
        source = mmm._link_spec.original_scale_transform(
            mmm.model[Var.CHANNEL_CONTRIBUTION], mmm.model[Var.TARGET_SCALE]
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
        prior = serialization.deserialize_model_config({"prior": spec["prior"]})[
            "prior"
        ]
        self._validate_component_support(mmm, spec["channels"], prior)
        source = self.frozen_graph
        if source is None:
            source = self._freeze_training_contribution_graph(mmm)
        # Resolve selections and the observation mask against the historical
        # labels, never the mutable prediction coordinates.
        indices = pd.Index(mmm.model_coords[Dim.CHANNEL]).get_indexer(spec["channels"])
        if self.mask is None:
            raise ValueError("Contributions require an observed training mask")
        mask = self.mask.reindex({dim: mmm.model_coords[dim] for dim in self.mask.dims})
        total = source.isel(channel=indices).where(as_xtensor(mask), 0).sum()
        denominator = float(mmm.xarray_dataset[DatasetVar.TARGET].where(mask, 0).sum())
        value = total / denominator if spec["quantity"] == "share" else total
        value_aggregate = mmm.model.replace_rvs_by_values([value])[0]
        if prior.distribution == "LogNormal" and is_identically_zero(
            value_aggregate.values
        ):
            raise ValueError(
                "LogNormal factors require a positive historical contribution; "
                "the fixed response is identically zero"
            )
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
        with mmm.model:
            pmd.Deterministic(spec["name"], value)
            pm.Potential(f"{spec['name']}_factor", logp)
        if self.frozen_graph is None:
            self.frozen_graph = source
            self.fingerprint = _training_fingerprint(mmm.xarray_dataset, self.mask)
        self.specs.append(copy.deepcopy(spec))

    def to_attrs(self) -> dict[str, str]:
        """Store small definitions and a fingerprint, with no numeric training arrays."""
        return {
            "contribution_factors": json.dumps(
                {Attr.VERSION: 1, "specs": self.specs, "fingerprint": self.fingerprint}
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
        specs = metadata.get("specs", [])
        if metadata and metadata.get(Attr.VERSION) != 1:
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
        config = {**mmm.sampler_config, **kwargs}
        if "model" in config or config.get("return_inferencedata") is False:
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
        # Reset standard MMM inputs on a clone. User-added RVs, Data, and
        # Deterministics remain intact; fitted results and configuration are shared
        # read-only, with no rebuild or deep copy of the MMM needed.
        sampling_copy = clone_model(mmm.model)
        mmm._set_xarray_data(mmm.xarray_dataset, model=sampling_copy)
        for effect in mmm.mu_effects:
            effect.set_data(mmm, sampling_copy, mmm.xarray_dataset)
        with sampling_copy:
            pm.set_data(self.inputs)
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
            result = pm.sample(**config)
        result["prior"] = result.posterior.dataset
        result = result.drop_nodes("posterior")
        if "sample_stats" in result:
            result["sample_stats_prior"] = result.sample_stats.dataset
            result = result.drop_nodes("sample_stats")
        result.attrs["sampling_target"] = (
            "parameter priors multiplied by historical contribution factors"
        )
        return result
