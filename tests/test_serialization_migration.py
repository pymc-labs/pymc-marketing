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
"""Tests for the migration tool (v0 -> v1 format conversion)."""

import json
import subprocess
import sys

import arviz as az
import numpy as np
import xarray as xr

from pymc_marketing.serialization_migration import CURRENT_VERSION, migrate_idata


def _make_v0_idata():
    """Create a minimal InferenceData with v0 (old-format) attrs."""
    posterior = xr.Dataset({"x": xr.DataArray(np.random.randn(4, 100))})
    idata = az.InferenceData(posterior=posterior)
    idata.attrs["id"] = "abc123"
    idata.attrs["model_type"] = "MMM"
    idata.attrs["version"] = "0.0.1"
    idata.attrs["sampler_config"] = json.dumps({})
    idata.attrs["model_config"] = json.dumps({})
    idata.attrs["adstock"] = json.dumps(
        {
            "lookup_name": "geometric",
            "l_max": 4,
            "normalize": True,
            "mode": "After",
            "priors": {
                "alpha": {"distribution": "Beta", "dims": ["channel"], "mu": 0.5}
            },
        }
    )
    idata.attrs["saturation"] = json.dumps(
        {
            "lookup_name": "logistic",
            "priors": {"lam": {"distribution": "Gamma", "mu": 1.0}},
        }
    )
    idata.attrs["time_varying_intercept"] = json.dumps(
        {
            "hsgp_class": "HSGP",
            "m": 200,
            "L": None,
            "eta": 1.0,
            "ls": 5.0,
            "dims": ["date"],
        }
    )
    idata.attrs["mu_effects"] = json.dumps(
        [
            {
                "class": "FourierEffect",
                "fourier": {
                    "__type__": "pymc_marketing.mmm.fourier.YearlyFourier",
                    "n_order": 2,
                },
            },
            {
                "class": "LinearTrendEffect",
                "trend": {"n_changepoints": 5},
                "prefix": "trend",
            },
        ]
    )
    return idata


class TestMigrateIdataV0ToV1:
    """Verify v0 -> v1 migration rewrites attrs correctly."""

    def test_migration_adds_serialization_version(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        assert migrated.attrs["__serialization_version__"] == str(CURRENT_VERSION)

    def test_migration_converts_adstock_lookup_name_to_type(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        adstock = json.loads(migrated.attrs["adstock"])
        assert "__type__" in adstock
        assert "lookup_name" not in adstock
        assert (
            adstock["__type__"]
            == "pymc_marketing.mmm.components.adstock.GeometricAdstock"
        )

    def test_migration_converts_saturation_lookup_name_to_type(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        sat = json.loads(migrated.attrs["saturation"])
        assert "__type__" in sat
        assert "lookup_name" not in sat
        assert (
            sat["__type__"]
            == "pymc_marketing.mmm.components.saturation.LogisticSaturation"
        )

    def test_migration_converts_hsgp_class_to_type(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        tvi = json.loads(migrated.attrs["time_varying_intercept"])
        assert "__type__" in tvi
        assert "hsgp_class" not in tvi
        assert tvi["__type__"] == "pymc_marketing.mmm.hsgp.HSGP"

    def test_migration_converts_mu_effect_class_to_type(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        effects = json.loads(migrated.attrs["mu_effects"])
        for effect in effects:
            assert "__type__" in effect
            assert "class" not in effect

    def test_migration_drops_stale_id(self):
        idata = _make_v0_idata()
        migrated = migrate_idata(idata)
        assert "id" not in migrated.attrs

    def test_already_v1_is_noop(self):
        idata = _make_v0_idata()
        idata.attrs["__serialization_version__"] = str(CURRENT_VERSION)
        original_attrs = dict(idata.attrs)
        migrated = migrate_idata(idata)
        assert migrated.attrs == original_attrs


class TestMigrateIdataCLI:
    """Verify the CLI entry point."""

    def test_cli_migrates_file(self, tmp_path):
        """python -m pymc_marketing.serialization_migration <file> should migrate in-place."""
        posterior = xr.Dataset({"x": xr.DataArray(np.random.randn(4, 100))})
        idata = az.InferenceData(posterior=posterior)
        idata.attrs["model_type"] = "MMM"
        idata.attrs["version"] = "0.0.1"
        idata.attrs["adstock"] = json.dumps({"lookup_name": "geometric", "l_max": 4})
        fname = tmp_path / "model.nc"
        idata.to_netcdf(str(fname))

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "pymc_marketing.serialization_migration",
                str(fname),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        reloaded = az.from_netcdf(fname)
        assert "__serialization_version__" in reloaded.attrs
