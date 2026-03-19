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
"""Migration tool for old-format serialized models.

Converts v0 (pre-TypeRegistry) idata attrs to v1 (__type__-based) format.

Usage:
    python -m pymc_marketing.serialization_migration model.nc
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from typing import Any

import arviz as az

CURRENT_VERSION = 1

_ADSTOCK_TYPE_MAP = {
    "geometric": "pymc_marketing.mmm.components.adstock.GeometricAdstock",
    "delayed": "pymc_marketing.mmm.components.adstock.DelayedAdstock",
    "weibull_cdf": "pymc_marketing.mmm.components.adstock.WeibullCDFAdstock",
    "weibull_pdf": "pymc_marketing.mmm.components.adstock.WeibullPDFAdstock",
    "binomial": "pymc_marketing.mmm.components.adstock.BinomialAdstock",
    "no_adstock": "pymc_marketing.mmm.components.adstock.NoAdstock",
}

_SATURATION_TYPE_MAP = {
    "logistic": "pymc_marketing.mmm.components.saturation.LogisticSaturation",
    "tanh": "pymc_marketing.mmm.components.saturation.TanhSaturation",
    "tanh_baselined": "pymc_marketing.mmm.components.saturation.TanhSaturationBaselined",
    "michaelis_menten": "pymc_marketing.mmm.components.saturation.MichaelisMentenSaturation",
    "hill": "pymc_marketing.mmm.components.saturation.HillSaturation",
    "hill_sigmoid": "pymc_marketing.mmm.components.saturation.HillSaturationSigmoid",
    "inverse_scaled_logistic": "pymc_marketing.mmm.components.saturation.InverseScaledLogisticSaturation",
    "root": "pymc_marketing.mmm.components.saturation.RootSaturation",
    "no_saturation": "pymc_marketing.mmm.components.saturation.NoSaturation",
}

_HSGP_CLASS_MAP = {
    "HSGP": "pymc_marketing.mmm.hsgp.HSGP",
    "SoftPlusHSGP": "pymc_marketing.mmm.hsgp.SoftPlusHSGP",
    "HSGPPeriodic": "pymc_marketing.mmm.hsgp.HSGPPeriodic",
}

_MUEFFECT_CLASS_MAP = {
    "FourierEffect": "pymc_marketing.mmm.additive_effect.FourierEffect",
    "LinearTrendEffect": "pymc_marketing.mmm.additive_effect.LinearTrendEffect",
    "EventAdditiveEffect": "pymc_marketing.mmm.additive_effect.EventAdditiveEffect",
}


def _migrate_v0_to_v1(attrs: dict[str, Any]) -> dict[str, Any]:
    """Rewrite v0 attrs to v1 format."""
    attrs = dict(attrs)

    if "adstock" in attrs:
        adstock = json.loads(attrs["adstock"])
        if (
            isinstance(adstock, dict)
            and "lookup_name" in adstock
            and "__type__" not in adstock
        ):
            lookup = adstock.pop("lookup_name")
            adstock["__type__"] = _ADSTOCK_TYPE_MAP.get(lookup, lookup)
            attrs["adstock"] = json.dumps(adstock)

    if "saturation" in attrs:
        sat = json.loads(attrs["saturation"])
        if isinstance(sat, dict) and "lookup_name" in sat and "__type__" not in sat:
            lookup = sat.pop("lookup_name")
            sat["__type__"] = _SATURATION_TYPE_MAP.get(lookup, lookup)
            attrs["saturation"] = json.dumps(sat)

    for key in ("time_varying_intercept", "time_varying_media"):
        if key in attrs:
            data = json.loads(attrs[key])
            if (
                isinstance(data, dict)
                and "hsgp_class" in data
                and "__type__" not in data
            ):
                cls_name = data.pop("hsgp_class")
                data["__type__"] = _HSGP_CLASS_MAP.get(cls_name, cls_name)
                attrs[key] = json.dumps(data)

    if "mu_effects" in attrs:
        effects = json.loads(attrs["mu_effects"])
        for effect in effects:
            if (
                isinstance(effect, dict)
                and "class" in effect
                and "__type__" not in effect
            ):
                cls_name = effect.pop("class")
                effect["__type__"] = _MUEFFECT_CLASS_MAP.get(cls_name, cls_name)
        attrs["mu_effects"] = json.dumps(effects)

    attrs.pop("id", None)
    attrs["__serialization_version__"] = str(CURRENT_VERSION)

    return attrs


def migrate_idata(idata: az.InferenceData) -> az.InferenceData:
    """Migrate InferenceData attrs from old format to current version.

    Parameters
    ----------
    idata : az.InferenceData
        The InferenceData to migrate. Modified in-place and returned.

    Returns
    -------
    az.InferenceData
        The same object, with attrs updated to current version.
    """
    version_str = idata.attrs.get("__serialization_version__", "0")
    version = int(version_str)

    if version >= CURRENT_VERSION:
        return idata

    migrations = {0: _migrate_v0_to_v1}

    while version < CURRENT_VERSION:
        if version not in migrations:
            raise ValueError(
                f"No migration path from version {version} to {version + 1}"
            )
        idata.attrs = migrations[version](idata.attrs)
        version += 1

    return idata


def main() -> None:
    """CLI entry point: ``python -m pymc_marketing.serialization_migration <file.nc>``."""
    if len(sys.argv) != 2:
        print("Usage: python -m pymc_marketing.serialization_migration <model.nc>")
        sys.exit(1)

    fname = sys.argv[1]
    print(f"Loading {fname}...")
    idata = az.from_netcdf(fname)

    version = idata.attrs.get("__serialization_version__", "0")
    if int(version) >= CURRENT_VERSION:
        print(f"Already at version {version}, nothing to do.")
        return

    print(f"Migrating from v{version} to v{CURRENT_VERSION}...")
    migrate_idata(idata)

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    idata.to_netcdf(tmp_path)
    shutil.move(tmp_path, fname)
    print(f"Done. Saved migrated model to {fname}")


if __name__ == "__main__":
    main()
