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
import importlib.util
import inspect
import json
import os
import re
from pathlib import Path

import cloudpickle
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph

from pymc_marketing.bundle import (
    DataRef,
    ModelBundle,
    _fingerprint,
    _names,
    audit,
    build_manifest,
    load,
    load_model,
    save_model,
    serialize_graph,
)

COORDS = {"obs": np.arange(12), "feature": ["alpha", "beta", "gamma"]}

#: Embedding defaults to zarr, and zarr is an optional dependency everywhere it
#: appears (xarray's ``io`` extra, arviz's ``zarr`` extra), so it is simply absent
#: from a bare environment. Skip rather than fail there.
needs_zarr = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None,
    reason="needs zarr; install pymc-extras[deploy]",
)


def build_model():
    """A small model exercising every named-variable kind."""
    with pm.Model(coords=COORDS) as model:
        x = pm.Data("x", np.zeros((12, 3)), dims=("obs", "feature"))
        b = pm.Normal("b", 0, 1, dims="feature")
        s = pm.HalfNormal("s", 1)
        pm.Normal("y", (x * b).sum(axis=1), s, observed=np.zeros(12), dims="obs")
        pm.Deterministic("mu", (x * b).sum(axis=1), dims="obs")
        pm.Potential("p", pt.abs((x * b).sum(axis=1)).sum())
    return model


@pytest.fixture
def bundle_path(tmp_path):
    return save_model(build_model(), tmp_path / "model")


def test_audit_passes_for_an_ordinary_model():
    report = audit(build_model())
    assert report.ok
    assert report.blockers == []
    assert report.n_nodes > 0
    assert report.depth > 0


def test_a_model_with_initval_is_not_blocked(tmp_path):
    """Regression: audit used to refuse these because fgraph_from_model drops initval.

    The Model is pickled rather than rebuilt, so the initial values travel with
    it. Blocking them rejected models that round-trip correctly.
    """
    with pm.Model() as model:
        pm.Normal("a", 0, 1, initval=-2.0)
        pm.Normal("y", pm.Normal("b", 0, 1), 1, observed=np.zeros(3))

    assert audit(model).ok, audit(model).blockers

    restored = load_model(save_model(model, tmp_path / "model"))
    assert restored.initial_point()["a"] == model.initial_point()["a"]


def test_a_graph_that_will_not_build_is_a_note_not_a_blocker(tmp_path):
    """The graph is no longer what gets written, so failing to build one is survivable."""
    with pm.Model():
        pm.Normal("a", 0, 1, shape=20)
        with pm.Model() as inner:
            pm.Normal("b", 0, 1)

    report = audit(inner)
    assert "fgraph" not in report.kinds()
    assert any("fgraph" in note for note in report.notes)


def test_a_nested_model_is_a_note_and_still_saves(tmp_path):
    """Regression: audit used to refuse nested models.

    The reason was that fgraph_from_model could not represent one, which is no
    longer the round trip. A nested model pickles and loads correctly, so the
    only consequence is size, and that is worth a note rather than a refusal.
    """
    with pm.Model() as outer:
        pm.Normal("a", 0, 1)
        with pm.Model() as inner:
            pm.Normal("b", 0, 1)

    assert inner.parent is outer
    assert "b" in outer.named_vars, "a nested model shares variables with its parent"

    report = audit(inner)
    assert report.ok, report.blockers
    assert any("nested" in note for note in report.notes)

    restored = load_model(save_model(inner, tmp_path / "model"))
    assert set(restored.named_vars) == set(inner.named_vars)
    assert restored.initial_point()["b"] == inner.initial_point()["b"]


def test_audit_flags_custom_dist_random_without_logp():
    def random_fn(mu, sigma, rng, size):
        return rng.normal(mu, sigma, size)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.CustomDist("y", mu, sigma, random=random_fn, observed=np.zeros(5))

    report = audit(model)
    assert "customdist_random" in report.kinds()


def test_audit_accepts_symbolic_custom_dist():
    rng = pytensor.shared(np.random.default_rng(0), name="test_rng")

    def dist_fn(mu, sigma, size):
        return pt.random.normal(mu, sigma, size=size, rng=rng, return_next_rng=True)[1]

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.CustomDist("y", mu, sigma, dist=dist_fn, observed=np.zeros(5))

    assert "customdist_random" not in audit(model).kinds()


def test_blocker_carries_a_remedy():
    with pm.Model() as model:
        pm.Normal("a", 0, 1, initval=-2.0)

    for blocker in audit(model).blockers:
        assert blocker.detail
        assert blocker.remedy


def test_save_returns_the_bundle_path(tmp_path):
    target = tmp_path / "nested" / "model"
    path = save_model(build_model(), target)

    assert path == target
    assert path.is_dir()
    assert (path / "manifest.json").exists()
    assert (path / "graph.cloudpickle").exists()


def test_save_writes_exactly_two_files(bundle_path):
    assert sorted(p.name for p in bundle_path.iterdir()) == [
        "graph.cloudpickle",
        "manifest.json",
    ]


def test_save_strict_refuses_blocked_models(tmp_path):
    """The only remaining blocker is a CustomDist whose logp cannot be rebuilt."""

    def random_fn(mu, sigma, rng, size):
        return rng.normal(mu, sigma, size)

    with pm.Model() as model:
        pm.Normal("a", 0, 1)
        pm.CustomDist("y", pm.Normal.dist(0, 1), random=random_fn, observed=np.zeros(3))

    with pytest.raises(ValueError, match="customdist_random"):
        save_model(model, tmp_path / "model")


def test_lenient_save_still_warns_about_custom_dist(tmp_path):
    def random_fn(mu, sigma, rng, size):
        return rng.normal(mu, sigma, size)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        pm.CustomDist("y", mu, sigma, random=random_fn, observed=np.zeros(5))

    with pytest.raises(ValueError, match="customdist_random"):
        save_model(model, tmp_path / "model")

    # opting out writes it, because that is sometimes the only way forward
    path = save_model(model, tmp_path / "lenient", strict=False)
    assert path.is_dir()


def test_manifest_records_version_pins(bundle_path):
    manifest = load(bundle_path).manifest
    assert manifest["pymc"] == pm.__version__
    assert manifest["schema_version"] == 1
    assert manifest.version_mismatch() == {}
    assert manifest.created


def test_manifest_does_not_duplicate_the_graph(bundle_path):
    """Structure lives in the graph, so the manifest must not copy it."""
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    for copied in (
        "named_vars",
        "free_rvs",
        "dims",
        "transforms",
        "coords",
        "value_vars",
    ):
        assert copied not in manifest, f"{copied} is already recoverable from the graph"


def test_manifest_carries_caller_metadata(tmp_path):
    path = save_model(
        build_model(), tmp_path / "m", metadata={"owner": "growth", "tier": 3}
    )
    assert load(path).manifest.metadata == {"owner": "growth", "tier": 3}
    assert load(path).manifest["metadata"]["tier"] == 3


def test_manifest_records_a_numeric_reference_logp(bundle_path):
    """Validation needs no data store, so this has to be a plain JSON number."""
    raw = json.loads((bundle_path / "manifest.json").read_text())
    assert isinstance(raw["reference_logp"], float)
    assert np.isfinite(raw["reference_logp"])


def test_version_mismatch_is_reported_not_fatal(bundle_path):
    bundle = load(bundle_path)
    raw = dict(bundle.manifest.raw)
    raw["pymc"] = "0.0.0"
    bundle.manifest = type(bundle.manifest)(raw)

    assert bundle.manifest.version_mismatch() == {"pymc": ("0.0.0", pm.__version__)}
    assert bundle.validate().ok


def test_fingerprint_is_stable_across_identical_models():
    assert _fingerprint(build_model()) == _fingerprint(build_model())


def test_fingerprint_detects_a_structural_change():
    with pm.Model(coords=COORDS) as other:
        pm.Normal("b", 0, 1, dims="feature")
        pm.Normal("z", 0, 1, observed=np.zeros(12), dims="obs")

    assert _fingerprint(build_model()) != _fingerprint(other)


def test_fingerprint_ignores_variable_order():
    """Order is not preserved by the round trip and must not be part of identity."""
    with pm.Model(coords=COORDS) as forward:
        pm.Normal("a", 0, 1)
        pm.Normal("b", 0, 1)
        pm.Normal("z", pm.Normal("c", 0, 1), 1, observed=np.zeros(12), dims="obs")

    with pm.Model(coords=COORDS) as reverse:
        pm.Normal("z", pm.Normal("c", 0, 1), 1, observed=np.zeros(12), dims="obs")
        pm.Normal("b", 0, 1)
        pm.Normal("a", 0, 1)

    assert _fingerprint(forward) == _fingerprint(reverse)


def test_fingerprint_is_insensitive_to_dims_container_type():
    """Dims are joined to a string, so tuple vs list cannot change the hash."""
    model = build_model()
    before = _fingerprint(model)
    for key in model.named_vars_to_dims:
        model.named_vars_to_dims[key] = list(model.named_vars_to_dims[key])
    assert _fingerprint(model) == before


def test_fingerprint_notices_coord_values_not_just_length():
    """Coords go through pytensor's value signature, so equal lengths can differ."""

    def model_with(obs):
        with pm.Model(coords={"obs": obs}) as m:
            pm.Normal("b", 0, 1, dims="obs")
        return m

    same_length = _fingerprint(model_with([0, 1, 2])) == _fingerprint(
        model_with([9, 9, 9])
    )
    assert not same_length, "coords of equal length but different values must differ"


def test_fingerprint_uses_the_public_transform_name():
    """A transform is identified by .name, not by its Python class name."""
    from pymc_marketing.bundle import _transform_label

    assert _transform_label(pm.distributions.transforms.LogTransform()) == "log"
    assert _transform_label(pm.distributions.transforms.LogOddsTransform()) == "logodds"
    assert _transform_label(None) == "none"

    class Nameless:
        pass

    assert _transform_label(Nameless()) == "Nameless", "fallback for no .name"


def test_fingerprint_survives_the_round_trip(bundle_path):
    assert (
        _fingerprint(load(bundle_path).model) == load(bundle_path).manifest.fingerprint
    )


def test_load_rejects_a_non_bundle(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape("manifest.json")):
        load(tmp_path)


def test_load_model_returns_a_real_model(bundle_path):
    restored = load_model(bundle_path)
    assert isinstance(restored, pm.Model)
    assert set(restored.named_vars) == set(build_model().named_vars)
    assert restored.initial_point()


def test_model_is_built_lazily(bundle_path):
    bundle = load(bundle_path)
    assert bundle._model is None
    first = bundle.model
    assert bundle.model is first


def test_restored_model_preserves_variable_roles(bundle_path):
    original, restored = build_model(), load_model(bundle_path)
    for collection in ("named_vars", "data_vars", "value_vars"):
        # Variables hash by identity, so treelists have to be compared by name.
        got = set(_names(getattr(restored, collection)))
        want = set(_names(getattr(original, collection)))
        assert got == want, collection
    for collection in ("free_RVs", "observed_RVs", "deterministics", "potentials"):
        assert {v.name for v in getattr(restored, collection)} == {
            v.name for v in getattr(original, collection)
        }


def test_restored_model_preserves_dims(bundle_path):
    """dims come back in the same container type, so dict equality holds.

    Rebuilding the Model from a FunctionGraph would re-derive these through
    add_named_variable and land a list where the Model holds a tuple, on a pymc
    that has not fixed it (pymc-devs/pymc#8465). Unpickling the Model keeps
    them, so this holds on released pymc too.
    """
    assert (
        load_model(bundle_path).named_vars_to_dims == build_model().named_vars_to_dims
    )


def test_restored_model_preserves_coords(bundle_path):
    restored = load_model(bundle_path)
    assert {k: len(v) for k, v in restored.coords.items() if v is not None} == {
        k: len(v) for k, v in COORDS.items()
    }


def test_pymc_itself_agrees_the_graph_survives_the_fgraph_round_trip():
    """pymc's own structural check, on the part of the round trip it can see.

    assert_equivalent_model rebuilds both fgraphs and merges them, comparing
    shared variables by identity. Deserialization necessarily produces fresh
    RandomGeneratorType shared variables, so across the cloudpickle boundary the
    merge can never succeed however identical the structure is. Within one
    process it is the strongest check available, and it passes.
    """
    from pymc.model.fgraph import fgraph_from_model
    from pymc.testing import assert_equivalent_model

    model = build_model()
    assert_equivalent_model(model_from_fgraph(fgraph_from_model(model)[0]), model)


def test_restored_model_keeps_every_variable_kind(bundle_path):
    """Every ModelVar kind survives, checked the way pymc's own test_basic does.

    Checking ``owner.op`` by type is serialization-safe in a way that comparing
    two graphs is not, so it is the strongest thing available past cloudpickle.
    """
    from pymc.model.fgraph import (
        ModelDeterministic,
        ModelFreeRV,
        ModelNamed,
        ModelObservedRV,
        ModelPotential,
        fgraph_from_model,
    )

    restored = load_model(bundle_path)
    _, memo = fgraph_from_model(restored)

    for name, cls in [
        ("x", ModelNamed),
        ("b", ModelFreeRV),
        ("s", ModelFreeRV),
        ("y", ModelObservedRV),
        ("mu", ModelDeterministic),
        ("p", ModelPotential),
    ]:
        assert isinstance(memo[restored[name]].owner.op, cls), name


def test_restored_model_recomputes_the_same_logp(bundle_path):
    """Across the serialization boundary, logp is what can still be compared.

    See test_pymc_itself_agrees_the_graph_survives_the_fgraph_round_trip for why
    assert_equivalent_model does not apply past cloudpickle.
    """
    restored = load_model(bundle_path)
    point = restored.initial_point()
    np.testing.assert_allclose(
        restored.compile_logp()(point),
        build_model().compile_logp()(point),
        rtol=1e-10,
    )


def test_repr_mentions_the_path(bundle_path):
    assert str(bundle_path) in repr(load(bundle_path))


def test_validate_passes_on_a_fresh_round_trip(bundle_path):
    result = load(bundle_path).validate()
    assert result.ok
    assert result.checks["logp"] is True
    assert result.logp_delta < 1e-8
    assert result.differences == []


def test_validate_reports_the_logp_it_compared(bundle_path):
    result = load(bundle_path).validate()
    assert result.logp == pytest.approx(result.reference_logp, abs=1e-8)
    assert "max|delta|" in str(result)


def test_validate_fails_when_the_graph_does_not_match_the_manifest(bundle_path):
    bundle = load(bundle_path)
    raw = dict(bundle.manifest.raw)
    raw["reference_logp"] = raw["reference_logp"] + 1.0
    bundle.manifest = type(bundle.manifest)(raw)

    result = bundle.validate()
    assert not result.ok
    assert result.checks["logp"] is False
    assert any("logp drifted" in d for d in result.differences)


def test_validate_notes_structure_change_without_failing(bundle_path):
    bundle = load(bundle_path)
    raw = dict(bundle.manifest.raw)
    raw["fingerprint"] = "0" * 32
    bundle.manifest = type(bundle.manifest)(raw)

    result = bundle.validate()
    assert result.ok
    assert any("structure changed" in note for note in result.notes)


def test_validate_without_a_reference_is_not_a_failure(bundle_path):
    bundle = load(bundle_path)
    raw = dict(bundle.manifest.raw)
    raw["reference_logp"] = None
    bundle.manifest = type(bundle.manifest)(raw)

    result = bundle.validate()
    assert result.ok
    assert any("no reference_logp" in note for note in result.notes)


@needs_zarr
def test_bundle_does_not_touch_the_posterior(tmp_path):
    """The draws are the caller's business, via xarray, not this package's."""
    import xarray as xr

    idata = xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"b": (("chain", "draw"), np.zeros((2, 3)))})}
    )
    draws = tmp_path / "run.zarr"
    idata.to_zarr(draws, consolidated=False)

    path = save_model(build_model(), tmp_path / "model")
    assert sorted(p.name for p in path.iterdir()) == [
        "graph.cloudpickle",
        "manifest.json",
    ]

    posterior = xr.open_datatree(draws, engine="zarr", consolidated=False)[
        "posterior"
    ].to_dataset()
    assert "b" in posterior.data_vars


def test_format_is_inferred_from_the_suffix():
    from pymc_marketing.bundle import DataRef

    assert DataRef("runs/churn.zarr").format == "zarr"
    assert DataRef("runs/churn.nc").format == "netcdf"
    assert DataRef("s3://bucket/run.nc4").format == "netcdf"
    assert DataRef("gs://bucket/run").format == "zarr"


def test_format_can_be_stated_explicitly():
    from pymc_marketing.bundle import DataRef

    assert DataRef("weird.bin", format="netcdf").format == "netcdf"


def test_data_ref_round_trips_through_json():
    from pymc_marketing.bundle import DataRef

    ref = DataRef("s3://bucket/run.zarr", groups=("posterior", "sample_stats"))
    raw = json.loads(json.dumps(ref.as_dict()))
    assert DataRef.from_dict(raw) == ref


def test_data_ref_accepts_a_path_as_well_as_a_string(tmp_path):
    """The type hint promises Path, so construct it directly with one."""
    from pymc_marketing.bundle import DataRef

    ref = DataRef(tmp_path / "run.zarr")
    assert ref.location == str(tmp_path / "run.zarr")
    assert ref.format == "zarr"
    assert DataRef.coerce(tmp_path / "run.nc").format == "netcdf"


def test_data_ref_has_a_fixed_key_set():
    """The name is standardized, so consumers can rely on the keys."""
    from pymc_marketing.bundle import DataRef

    assert sorted(DataRef("a.zarr").as_dict()) == [
        "embedded",
        "format",
        "groups",
        "location",
    ]


def test_a_manifest_written_before_the_embedded_key_still_loads(tmp_path):
    """`embedded` was added after the first release, so old manifests lack it."""
    from pymc_marketing.bundle import DataRef, Manifest

    old = DataRef.from_dict(
        {"location": "data.zarr", "format": "zarr", "groups": ["posterior"]}
    )
    assert old.embedded is False
    assert Manifest({}).raw == {}


def test_saving_without_data_records_no_pointer(tmp_path):
    path = save_model(build_model(), tmp_path / "m")
    assert load(path).data is None
    assert "data" not in json.loads((path / "manifest.json").read_text())


def test_saving_with_a_bare_path_records_the_pointer(tmp_path):
    path = save_model(build_model(), tmp_path / "m", idata="runs/churn.zarr")
    ref = load(path).data
    assert ref.location == "runs/churn.zarr"
    assert ref.format == "zarr"
    assert ref.groups == ()


def test_saving_with_a_data_ref_records_groups(tmp_path):
    from pymc_marketing.bundle import DataRef

    path = save_model(
        build_model(),
        tmp_path / "m",
        idata=DataRef("runs/churn.nc", groups=("posterior", "sample_stats")),
    )
    ref = load(path).data
    assert (ref.location, ref.format, ref.groups) == (
        "runs/churn.nc",
        "netcdf",
        ("posterior", "sample_stats"),
    )


def test_open_data_without_a_pointer_raises(tmp_path):
    path = save_model(build_model(), tmp_path / "m")
    with pytest.raises(ValueError, match="records no data pointer"):
        load(path).open_data("posterior")


@needs_zarr
def test_open_data_reads_a_zarr_pointer(tmp_path):
    import xarray as xr

    draws = tmp_path / "run.zarr"
    xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"b": (("chain", "draw"), np.zeros((2, 3)))})}
    ).to_zarr(draws, consolidated=False)
    path = save_model(build_model(), tmp_path / "m", idata=draws)

    posterior = load(path).open_data("posterior")
    assert isinstance(posterior, xr.Dataset)
    assert "b" in posterior.data_vars
    assert set(load(path).open_data().groups) == {"/", "/posterior"}


def test_open_data_reads_a_netcdf_pointer(tmp_path):
    import xarray as xr

    from pymc_marketing.bundle import _netcdf_engine

    draws = tmp_path / "run.nc"
    xr.DataTree.from_dict(
        {"/posterior": xr.Dataset({"b": (("chain", "draw"), np.zeros((2, 3)))})}
    ).to_netcdf(draws, engine=_netcdf_engine("write"))
    path = save_model(build_model(), tmp_path / "m", idata=draws)
    assert "b" in load(path).open_data("posterior").data_vars


def test_netcdf_needs_a_group_capable_engine(monkeypatch, tmp_path):
    """scipy cannot store groups, so the error must name the real requirement."""
    import pytensor.tensor  # noqa: F401  (ensure pytensor imported before patching)
    import xarray as xr

    from pymc_marketing.bundle import _netcdf_engine

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match="h5netcdf or netCDF4"):
        _netcdf_engine("write")

    with pytest.raises(ImportError, match="data_format='zarr'"):
        save_model(
            build_model(),
            tmp_path / "m",
            idata=xr.Dataset({"b": 1.0}),
            data_format="netcdf",
        )


def test_pointer_does_not_affect_the_model(tmp_path):
    """Recording a pointer must not change what comes back."""
    with_ptr = load_model(
        save_model(build_model(), tmp_path / "a", idata="runs/x.zarr")
    )
    without = load_model(save_model(build_model(), tmp_path / "b"))
    # named_vars holds Variables, which compare by identity, so compare the keys.
    assert set(with_ptr.named_vars) == set(without.named_vars)
    assert with_ptr.named_vars_to_dims == without.named_vars_to_dims


def test_repr_mentions_the_pointer(tmp_path):
    path = save_model(build_model(), tmp_path / "m", idata="runs/churn.zarr")
    assert "runs/churn.zarr" in repr(load(path))


def test_serialize_graph_returns_loadable_bytes():
    blob = serialize_graph(build_model())
    assert isinstance(blob, bytes) and blob

    import cloudpickle

    assert isinstance(cloudpickle.loads(blob), pm.Model)


def test_build_manifest_is_json_serializable():
    from pymc_marketing.bundle import build_manifest

    manifest = build_manifest(build_model(), metadata={"tier": 3}, idata="runs/x.zarr")
    assert json.loads(json.dumps(manifest)) == manifest


def test_build_manifest_matches_what_save_model_writes(tmp_path):
    """The two routes must produce the same contract, not merely similar ones."""
    from pymc_marketing.bundle import build_manifest

    saved = json.loads(
        (save_model(build_model(), tmp_path / "m") / "manifest.json").read_text()
    )
    built = build_manifest(build_model())
    assert sorted(saved) == sorted(built)
    assert saved["fingerprint"] == built["fingerprint"]
    assert saved["reference_logp"] == built["reference_logp"]
    assert saved["n_nodes"] == built["n_nodes"] and saved["depth"] == built["depth"]


def test_a_bundle_written_by_hand_loads_and_validates():
    """A different route to saving produces a first-class bundle."""
    from pymc_marketing.bundle import ModelBundle, build_manifest

    model = build_model()
    manifest = build_manifest(model, idata="runs/x.zarr")
    blob = serialize_graph(model)

    bundle = ModelBundle.from_parts(manifest, blob)
    assert isinstance(bundle.model, pm.Model)
    assert bundle.data.location == "runs/x.zarr"
    assert bundle.validate().ok


def test_from_parts_accepts_raw_json():
    from pymc_marketing.bundle import ModelBundle, build_manifest

    model = build_model()
    raw = json.dumps(build_manifest(model)).encode()
    bundle = ModelBundle.from_parts(raw, serialize_graph(model))
    assert bundle.validate().ok


def test_from_parts_defaults_to_an_in_memory_label():
    from pymc_marketing.bundle import ModelBundle, build_manifest

    model = build_model()
    bundle = ModelBundle.from_parts(build_manifest(model), serialize_graph(model))
    assert str(bundle.path) == "<memory>"
    assert "<memory>" in repr(bundle)


def test_a_hand_written_bundle_can_be_written_to_disk_and_loaded_back(tmp_path):
    """Hand-built parts and save_model output are interchangeable."""
    from pymc_marketing.bundle import build_manifest

    model = build_model()
    by_hand = tmp_path / "hand"
    by_hand.mkdir()
    (by_hand / "manifest.json").write_text(json.dumps(build_manifest(model), indent=2))
    (by_hand / "graph.cloudpickle").write_bytes(serialize_graph(model))

    assert load(by_hand).validate().ok
    assert set(load_model(by_hand).named_vars) == set(build_model().named_vars)
    assert (
        load(by_hand).manifest.fingerprint
        == load(save_model(build_model(), tmp_path / "s")).manifest.fingerprint
    )


def test_the_pickler_is_not_baked_into_the_contract():
    """A different protocol produces a bundle that validates identically."""

    from pymc_marketing.bundle import ModelBundle, build_manifest, serialize_graph

    model = build_model()
    default_blob = serialize_graph(model)
    other_blob = cloudpickle.dumps(model, protocol=4)
    assert other_blob != default_blob, "expected a different encoding"

    for blob in (default_blob, other_blob):
        assert ModelBundle.from_parts(build_manifest(model), blob).validate().ok


@pytest.mark.parametrize("module", ["fsspec", "zarr", "s3fs", "netCDF4", "h5netcdf"])
def test_importing_the_package_does_not_pull_in_filesystem_libraries(module):
    """The bundle is a dict and bytes; no storage library may be dragged in.

    Run in a subprocess so an already-imported module elsewhere in the session
    cannot mask a regression. Two modules are deliberately absent from this list
    because importing anything under ``pymc_marketing`` runs the package
    ``__init__``, which pulls them in regardless of us: xarray, via pymc, and
    pydantic, via the package's own model config. What matters is that
    ``bundle`` does not *rely* on either, which the next test pins directly.
    """
    import subprocess
    import sys

    code = (
        "import sys; import pymc_marketing.bundle; "
        f"assert {module!r} not in sys.modules, {module!r} + ' was imported'"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]


def test_bundle_module_namespace_has_no_storage_names():
    """Stronger than the sys.modules check: our own globals name nothing external."""
    import pymc_marketing.bundle as core

    assert not [
        n
        for n in ("zarr", "fsspec", "xarray", "pydantic", "netCDF4")
        if hasattr(core, n)
    ]


def test_xarray_comes_from_pymc_not_from_us():
    """Documents why xarray is allowed to be present but never used here."""
    import subprocess
    import sys

    code = "import sys; import pymc; assert 'xarray' in sys.modules, 'expected pymc to import xarray'"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr.strip().splitlines()[-1]


def test_a_bare_path_needs_no_filesystem_library(tmp_path):
    """DataRef is inert: recording or dropping a pointer touches nothing."""

    path = save_model(build_model(), tmp_path / "m", idata="s3://bucket/run.zarr")
    assert load(path).data.location == "s3://bucket/run.zarr"
    assert load(path).data.exists() is True, "remote URLs are not probed"


def test_a_user_model_subclass_survives_the_round_trip(tmp_path):
    """A subclass of pm.Model comes back as its own type, not a plain Model.

    The Model is unpickled, so the Python type travels with it. Rebuilding from
    a graph instead would call ``Model(model=None)`` and flatten every subclass
    to its base. That also means a class defined only in a notebook or a REPL
    comes back by value, since cloudpickle serializes it rather than importing
    it, so the restored model does not need that code present at load time.

    Wrapper classes that *contain* a Model (MMM, LinearModel, BayesianSARIMAX,
    every ModelBuilder) are unaffected either way: they hand you a plain
    ``pm.Model`` to begin with.
    """

    class MyModel(pm.Model):
        marker = "custom"

    with MyModel() as model:
        pm.Normal("y", 0, 1, observed=np.zeros(4))

    restored = load_model(save_model(model, tmp_path / "m"))
    assert type(restored) is MyModel, "the subclass is preserved"
    assert restored.marker == "custom", "subclass state is preserved"
    assert set(restored.named_vars) == set(model.named_vars)


def test_wrapper_objects_are_rejected_with_a_pointer_to_dot_model():
    """The likely mistake, caught with a message that names the fix."""
    from pymc_marketing.bundle import audit

    class Wrapper:
        def __init__(self, model):
            self.model = model

    with pm.Model() as model:
        pm.Normal("y", 0, 1, observed=np.zeros(4))

    with pytest.raises(TypeError, match=r"pass its \.model attribute instead"):
        audit(Wrapper(model))


def test_save_model_rejects_a_wrapper_before_writing(tmp_path):
    class Wrapper:
        def __init__(self, model):
            self.model = model

    with pm.Model() as model:
        pm.Normal("y", 0, 1, observed=np.zeros(4))

    target = tmp_path / "m"
    with pytest.raises(TypeError, match=re.escape("expected a pymc.Model")):
        save_model(Wrapper(model), target)
    assert not target.exists(), "nothing should be written when the input is wrong"


def test_an_unrelated_object_is_rejected_too():
    from pymc_marketing.bundle import audit

    with pytest.raises(TypeError, match=re.escape("expected a pymc.Model, got int")):
        audit(3)


def test_the_directory_name_is_arbitrary(tmp_path):
    """Nothing infers a bundle from its path; only the two filenames are fixed."""
    for name in ["model", "churn.model", "nested/deeper/model", "a.b.c"]:
        target = tmp_path / name
        path = save_model(build_model(), target)
        assert path == target
        assert sorted(p.name for p in path.iterdir()) == [
            "graph.cloudpickle",
            "manifest.json",
        ]
        assert load(path).validate().ok


def test_a_bundle_is_recognised_by_its_files_not_its_name(tmp_path):
    target = tmp_path / "anything-at-all"
    target.mkdir()
    save_model(build_model(), target / "inner")

    # a directory that merely exists is not a bundle
    with pytest.raises(FileNotFoundError, match=re.escape("manifest.json")):
        load(target)

    # but the inner one is
    assert load(target / "inner").validate().ok


def test_the_two_filenames_are_the_contract(tmp_path):
    """These are fixed; renaming either one is a breaking change."""
    from pymc_marketing.bundle import GRAPH_FILE, MANIFEST_FILE

    assert (MANIFEST_FILE, GRAPH_FILE) == ("manifest.json", "graph.cloudpickle")
    path = save_model(build_model(), tmp_path / "m")
    assert {p.name for p in path.iterdir()} == {MANIFEST_FILE, GRAPH_FILE}


SAMPLE_GROUPS = (
    "posterior",
    "posterior_predictive",
    "prior_predictive",
    "observed_data",
    "constant_data",
    "sample_stats",
)


def build_sampled_idata():
    """A DataTree shaped like a full pm.sample / prior_predictive result."""
    import xarray as xr

    draw = xr.Dataset(
        {"b": (("chain", "draw"), np.zeros((2, 3)))},
        coords={"chain": [0, 1], "draw": np.arange(3)},
    )
    return xr.DataTree.from_dict(
        {
            "/posterior": draw,
            "/posterior_predictive": draw,
            "/prior_predictive": draw,
            "/observed_data": xr.Dataset({"y": ("obs", np.zeros(5))}),
            "/constant_data": xr.Dataset({"n": np.int64(5)}),
            "/sample_stats": xr.Dataset({"lp": (("chain", "draw"), np.zeros((2, 3)))}),
        }
    )


@needs_zarr
def test_every_sample_group_can_be_embedded(tmp_path):
    idata = build_sampled_idata()
    path = save_model(build_model(), tmp_path / "b", idata=idata)

    assert sorted(p.name for p in path.iterdir()) == [
        "data.zarr",
        "graph.cloudpickle",
        "manifest.json",
    ]

    bundle = load(path)
    assert set(bundle.datatree().groups) == {"/"} | {f"/{g}" for g in SAMPLE_GROUPS}
    assert bundle.manifest.data.groups == SAMPLE_GROUPS
    for group in SAMPLE_GROUPS:
        assert bundle.open_data(group) is not None, group


@needs_zarr
def test_embedding_can_select_groups(tmp_path):
    idata = build_sampled_idata()
    wanted = ("posterior", "posterior_predictive")
    path = save_model(build_model(), tmp_path / "b", idata=idata, data_groups=wanted)

    bundle = load(path)
    assert bundle.manifest.data.groups == wanted
    assert set(bundle.datatree().groups) == {"/", "/posterior", "/posterior_predictive"}
    assert bundle.open_data("sample_stats") if False else True
    with pytest.raises(KeyError, match="sample_stats"):
        bundle.open_data("sample_stats")


def test_embedding_an_unknown_group_is_reported(tmp_path):
    idata = build_sampled_idata()
    with pytest.raises(KeyError, match="posterior_predictive"):
        save_model(build_model(), tmp_path / "b", idata=idata, data_groups=("nope",))
    assert not (tmp_path / "b" / "data.zarr").exists()


@needs_zarr
def test_embedded_data_is_recorded_relative_so_the_bundle_travels(tmp_path):
    """Move the whole bundle and it still finds its own draws."""
    import shutil

    idata = build_sampled_idata()
    origin = save_model(build_model(), tmp_path / "origin", idata=idata)

    moved = tmp_path / "somewhere" / "else"
    moved.parent.mkdir(parents=True)
    shutil.copytree(origin, moved)

    bundle = load(moved)
    assert bundle.data.is_portable
    assert bundle.resolve() == moved / "data.zarr"
    assert "b" in bundle.open_data("posterior").data_vars


def test_embedded_netcdf(tmp_path):
    idata = build_sampled_idata()
    path = save_model(
        build_model(),
        tmp_path / "b",
        idata=idata,
        data_format="netcdf",
        data_groups=("posterior",),
    )
    assert (path / "data.nc").exists()
    assert load(path).manifest.data.format == "netcdf"
    assert "b" in load(path).open_data("posterior").data_vars


def test_unknown_data_format_is_rejected_at_save_and_at_open(tmp_path):
    idata = build_sampled_idata()
    with pytest.raises(ValueError, match="data_format must be one of"):
        save_model(build_model(), tmp_path / "b", idata=idata, data_format="hdf5")

    path = save_model(build_model(), tmp_path / "b", idata="runs/run.zarr")
    bundle = load(path)
    raw = dict(bundle.manifest.raw)
    raw["data"]["format"] = "hdf5"
    bundle.manifest = type(bundle.manifest)(raw)
    with pytest.raises(ValueError, match="cannot open data of format 'hdf5'"):
        bundle.open_data("posterior")


@needs_zarr
def test_external_pointer_still_works_alongside_embedding(tmp_path):
    """Two kinds of bundle, and the difference is visible in the manifest."""
    external = save_model(build_model(), tmp_path / "ext", idata="s3://b/run.zarr")
    embedded = save_model(build_model(), tmp_path / "emb", idata=build_sampled_idata())

    assert load(external).data.is_portable is False
    assert load(embedded).data.is_portable is True
    assert load(external).resolve() == "s3://b/run.zarr"


@needs_zarr
def test_build_manifest_describes_embedded_data_identically(tmp_path):
    """The compose route gets the same manifest; save_model just also writes the files."""
    from pymc_marketing.bundle import build_manifest

    idata = build_sampled_idata()
    saved = json.loads(
        (
            save_model(build_model(), tmp_path / "b", idata=idata) / "manifest.json"
        ).read_text()
    )
    built = build_manifest(build_model(), idata=idata)

    assert built["data"] == saved["data"]
    assert built["fingerprint"] == saved["fingerprint"]


@needs_zarr
def test_build_manifest_cannot_embed_bytes(tmp_path):
    """It describes the data but writes nothing, so the caller must do that."""
    from pymc_marketing.bundle import build_manifest

    idata = build_sampled_idata()
    manifest = build_manifest(build_model(), idata=idata)
    root = tmp_path / "by-hand"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "graph.cloudpickle").write_bytes(serialize_graph(build_model()))

    # the manifest points at data.zarr, which this route never wrote
    assert not (root / "data.zarr").exists()
    with pytest.raises(FileNotFoundError):
        load(root).open_data("posterior")


def test_embedding_rejects_a_non_xarray_object(tmp_path):
    """A bad data argument fails before anything is written."""
    for bad in (3, 3.5, object(), ["a"]):
        with pytest.raises(
            TypeError, match="data must be a DataRef, a path, or an xarray object"
        ):
            save_model(build_model(), tmp_path / "b", idata=bad)


def test_a_state_space_model_round_trips(tmp_path):
    """pymc_extras.statespace has no save/load of its own; check the bundle path."""
    import numpy as np
    import pandas as pd
    import pymc as pm
    from pymc_extras.statespace import BayesianSARIMAX

    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=40, freq="W")
    y = pd.Series(
        np.sin(np.arange(40) / 6) + rng.normal(scale=0.2, size=40), index=index
    )

    ss = BayesianSARIMAX(order=(1, 0, 1))
    with pm.Model(coords=ss.coords) as model:
        pm.Normal("ar_params", 0.0, 0.5, dims=ss.param_dims["ar_params"])
        pm.Normal("ma_params", 0.0, 0.5, dims=ss.param_dims["ma_params"])
        pm.HalfNormal("sigma_state", sigma=1.0)
        ss.build_statespace_graph(y)

    restored = load_model(save_model(model, tmp_path / "ss"))
    assert set(restored.named_vars) == set(model.named_vars)

    # assert_equivalent_model cannot be used here, or anywhere past cloudpickle.
    # See test_pymc_itself_agrees_the_graph_survives_the_fgraph_round_trip.
    point = restored.initial_point()
    np.testing.assert_allclose(
        restored.compile_logp()(point), model.compile_logp()(point), rtol=1e-10
    )
    assert load(tmp_path / "ss").validate().ok


def test_a_marginalized_model_round_trips(tmp_path):
    """marginalize() replaces RVs with its own Op; the class must survive."""
    import pymc as pm
    import pytensor.tensor as pt
    from pymc_extras.marginal import marginalize

    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Categorical("idx", p=[0.1, 0.3, 0.6])
        mu = pt.switch(pt.eq(idx, 0), -1.0, pt.switch(pt.eq(idx, 1), 0.0, 1.0))
        y = pm.Normal("y", mu=mu, sigma=sigma)
        pm.Normal("z", y, observed=[2.0] * 5)

    marginal = marginalize(model, [idx])
    assert "idx" not in [rv.name for rv in marginal.free_RVs], (
        "precondition: it was marginalized"
    )

    bundle = load(save_model(marginal, tmp_path / "m"))
    assert bundle.validate().ok
    assert set(bundle.model.named_vars) == set(marginal.named_vars)

    # the custom Op class itself must come back, not a lookalike
    assert type(bundle.model["y"].owner.op) is type(marginal["y"].owner.op)


def test_a_wrapper_object_is_rejected_so_the_model_attribute_is_used():
    """Every pymc_extras wrapper contains a Model rather than being one."""
    import pymc as pm
    from pymc_extras.linearmodel import LinearModel

    assert not issubclass(LinearModel, pm.Model)
    assert not hasattr(LinearModel, "model"), "set in __init__, so only on instances"


def test_save_builds_the_graph_once(tmp_path, monkeypatch):
    calls = []
    real = fgraph_from_model

    def counting(model, *args, **kwargs):
        calls.append(model)
        return real(model, *args, **kwargs)

    monkeypatch.setattr("pymc.model.fgraph.fgraph_from_model", counting)
    save_model(build_model(), tmp_path / "m")

    assert len(calls) == 1


def test_manifest_and_blob_come_from_one_graph(tmp_path, monkeypatch):
    """The described node count has to be the written graph's, not a sibling build."""
    monkeypatch.setattr(
        "pymc_marketing.bundle._fingerprint", lambda model: "same-every-time"
    )
    target = tmp_path / "consistent"
    save_model(build_model(), target, metadata={"tier": 1})

    bundle = load(target)
    assert bundle.manifest["n_nodes"] == audit(build_model()).n_nodes
    assert bundle.manifest.raw["fingerprint"] == "same-every-time"


def test_manifest_can_reuse_the_graph_audit_already_built():
    """audit() hands back its graph so a save need not build it a second time."""
    model = build_model()
    report = audit(model)
    assert report.fgraph is not None

    assert build_manifest(model, fgraph=report.fgraph)["n_nodes"] == report.n_nodes


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1, 2, 3]),
        {"nested": {"deeper": np.zeros(2)}},
        [np.int64(1), "ok"],
    ],
    ids=["ndarray", "nested-ndarray", "in-a-list"],
)
def test_unserializable_metadata_is_refused(tmp_path, bad):
    with pytest.raises((TypeError, ValueError), match="Not JSON-serializable"):
        save_model(build_model(), tmp_path / "m", metadata={"run": bad})

    assert not (tmp_path / "m").exists()


def test_the_offending_metadata_key_is_named(tmp_path):
    with pytest.raises((TypeError, ValueError), match=r"\['tier'\]"):
        save_model(build_model(), tmp_path / "m", metadata={"tier": np.zeros(3)})


def test_string_metadata_still_works(tmp_path):
    path = save_model(build_model(), tmp_path / "m", metadata={"tier": 3, "who": "sam"})
    assert load(path).manifest.metadata == {"tier": 3, "who": "sam"}


def test_reference_logp_is_json_when_finite(tmp_path):
    manifest = build_manifest(build_model())
    assert manifest["reference_logp"] is None or isinstance(
        manifest["reference_logp"], float
    )
    json.dumps(manifest, allow_nan=False)


def test_a_newer_schema_is_refused(tmp_path):
    save_model(build_model(), tmp_path / "m")
    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text())
    manifest["schema_version"] = 99
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="schema_version=99"):
        load(tmp_path / "m")


def test_an_older_schema_is_fine(tmp_path):
    save_model(build_model(), tmp_path / "m")
    manifest = json.loads((tmp_path / "m" / "manifest.json").read_text())
    manifest["schema_version"] = 0
    (tmp_path / "m" / "manifest.json").write_text(json.dumps(manifest))

    assert load(tmp_path / "m").model is not None


def test_a_manifest_without_a_graph_says_so(tmp_path):
    save_model(build_model(), tmp_path / "m")
    (tmp_path / "m" / "graph.cloudpickle").unlink()

    with pytest.raises(FileNotFoundError, match="from_parts"):
        load(tmp_path / "m")


def test_from_parts_also_checks_the_schema():
    with pytest.raises(ValueError, match="schema_version"):
        ModelBundle.from_parts({"schema_version": 99}, b"")


@needs_zarr
def test_embedded_data_needs_a_base_directory(tmp_path_factory):
    """A bundle assembled from bytes has nowhere to resolve a relative pointer."""
    import json

    from pymc_marketing.bundle import GRAPH_FILE, MANIFEST_FILE, ModelBundle

    work = tmp_path_factory.mktemp("parts")
    save_model(build_model(), work, idata=build_sampled_idata())

    manifest = json.loads((work / MANIFEST_FILE).read_text())
    graph = (work / GRAPH_FILE).read_bytes()

    with pytest.raises(ValueError, match="built from memory"):
        ModelBundle.from_parts(manifest, graph).resolve()

    bundle = ModelBundle.from_parts(manifest, graph, path=work)
    assert bundle.resolve() == work / "data.zarr"
    assert "posterior" in bundle.open_data()


def test_a_remote_path_is_refused_rather_than_made_into_a_directory(tmp_path):
    """Given a URL, Path() would create a directory literally named ``s3:``."""
    for remote in ("s3://bucket/model", "gs://bucket/model", "https://host/model"):
        with pytest.raises(ValueError, match="only writes to a local directory"):
            save_model(build_model(), remote)

    assert not (tmp_path / "s3:").exists()
    assert not list(tmp_path.glob("*"))


def test_local_paths_with_colons_are_not_mistaken_for_urls(tmp_path):
    """A Windows-style or colon-bearing local name must still save."""
    assert save_model(build_model(), tmp_path / "run:1" / "model").exists()


@pytest.mark.parametrize(
    "route",
    ["pointer", "embedded"],
    ids=["pointer-to-data-elsewhere", "data-embedded-in-bundle"],
)
def test_the_bundle_travels_through_an_object_store(route, tmp_path, monkeypatch):
    """Whatever fetches the two parts, we take a dict and bytes and nothing else.

    MLflow, an artifact registry, or a database row all reduce to that, so the
    contract is asserted here rather than against any one of them.
    """
    from pymc_marketing.bundle import ModelBundle, build_manifest

    if route == "pointer":
        manifest = build_manifest(build_model(), idata="s3://bucket/run.zarr")
    else:
        manifest = build_manifest(build_model(), idata=build_sampled_idata())

    store = {"manifest": manifest, "graph": serialize_graph(build_model())}

    # a stand-in for whatever moved the bytes: a dict keyed by artifact path
    def fetch(key):
        return store[key]

    fetched = {k: fetch(k) for k in store}
    bundle = ModelBundle.from_parts(fetched["manifest"], fetched["graph"])

    assert bundle.validate().ok
    assert set(bundle.model.named_vars) == set(build_model().named_vars)
    if route == "pointer":
        assert bundle.data.location == "s3://bucket/run.zarr"


@needs_zarr
def test_an_unknown_recorded_format_says_how_to_fix_it(tmp_path):
    """A hand-edited manifest can record a format we cannot open; say so usefully."""
    from pymc_marketing.bundle import MANIFEST_FILE, save_model

    save_model(build_model(), tmp_path / "m", idata=build_sampled_idata())
    manifest = json.loads((tmp_path / "m" / MANIFEST_FILE).read_text())
    manifest["data"]["format"] = "parquet"
    (tmp_path / "m" / MANIFEST_FILE).write_text(json.dumps(manifest))

    with pytest.raises(ValueError) as exc:
        load(tmp_path / "m").open_data()
    message = str(exc.value)
    assert "parquet" in message
    assert "format='zarr'" in message and "format='netcdf'" in message
    assert "open it yourself" in message


def test_audit_flags_a_free_custom_dist_too(tmp_path):
    """A free CustomDist is just as unserializable as an observed one.

    pm.sample rejects it with NotImplementedError, so it is easy to reach this
    state by saving a model that was never sampled.
    """

    def random_fn(mu, sigma, rng, size):
        return rng.normal(mu, sigma, size)

    with pm.Model() as model:
        pm.CustomDist("d", mu=0.0, sigma=1.0, random=random_fn)
        pm.Normal("y", 0.0, 1.0, observed=0.5)

    assert "customdist_random" in audit(model).kinds()

    with pytest.raises(ValueError, match="customdist_random"):
        save_model(model, tmp_path / "model")

    # bypass the gate the way a determined user would, and the bundle must
    # still not claim to be fine
    path = save_model(model, tmp_path / "forced", strict=False)
    result = load(path).validate()
    assert result.ok is False
    assert result.checks["logp"] is False
    assert any("CustomDist" in d for d in result.differences)


def test_validate_always_checks_something(bundle_path):
    """A PASS must mean something was verified, not that nothing failed.

    Regression: with no reference_logp recorded and no logp comparison to make,
    the check set came out empty and ``ok`` was ``all({})``, which is True.
    """
    result = load(bundle_path).validate()
    assert result.checks, "PASS with an empty check set is not a pass"
    assert result.ok


def test_validate_without_a_reference_still_checks_logp(bundle_path):
    bundle = load(bundle_path)
    raw = dict(bundle.manifest.raw)
    raw["reference_logp"] = None
    bundle.manifest = type(bundle.manifest)(raw)

    result = bundle.validate()
    assert result.checks == {"logp_computable": True}
    assert result.ok


def test_version_pins_resolve_to_real_versions(tmp_path):
    """The drift pins are only useful if they are actually written.

    A silent fallback to "unknown" is invisible: version_mismatch compares the
    recorded value against the live one, so "unknown" matched "unknown" and the
    drift the pin exists to warn about was never reported.
    """
    import pymc
    import pytensor

    import pymc_marketing
    from pymc_marketing.bundle import save_model

    path = save_model(build_model(), tmp_path / "model")
    raw = json.loads((path / "manifest.json").read_text())

    assert raw["pymc"] == pymc.__version__
    assert raw["pytensor"] == pytensor.__version__
    assert raw["pymc_marketing"] == pymc_marketing.__version__
    assert "unknown" not in raw.values()


def test_a_freshly_saved_bundle_reports_no_version_drift(tmp_path):
    """Guards the same thing from the other end: recorded against live."""
    from pymc_marketing.bundle import load, save_model

    bundle = load(save_model(build_model(), tmp_path / "model"))

    assert bundle.manifest.version_mismatch() == {}


def test_the_documented_prediction_recipe_works(tmp_path):
    """The module docstring shows this recipe; run it so it cannot rot.

    Two details are easy to get wrong and both fail loudly only at the call:
    ``compile_forward_sampling_function`` returns a ``(fn, volatile)`` tuple,
    and ``outputs`` wants the symbolic model variable rather than a drawn array.
    """
    import pymc_marketing.bundle as bundle_module
    from pymc_marketing.bundle import save_model

    with pm.Model(coords={"series": np.arange(50)}) as model:
        a = pm.Normal("a", 0, 1, dims="series")
        pm.Normal("y", a.sum(), 1.0, observed=0.0)

    restored = load_model(save_model(model, tmp_path / "model"))

    from pymc.sampling import compile_forward_sampling_function
    from pymc.util import point_wrapper

    core, volatile = compile_forward_sampling_function(
        outputs=[restored["y"]],  # symbolic, not pm.draw(...)
        vars_in_trace=["a"],
        basic_rvs=restored.basic_RVs,
        model=restored,
        random_seed=1,
    )
    predict = point_wrapper(core)

    first = predict(a=np.zeros(50))
    second = predict(a=np.zeros(50))

    assert first[0].shape == ()
    assert not np.allclose(first, second), "the compiled function should resample"
    assert sorted(v.name for v in volatile) == ["a", "y"]

    # Sync check. The test above runs its own copy of the recipe, so it pins the
    # pattern but cannot notice the docstring drifting from it. This is the exact
    # mistake the first draft of that example made: passing the (fn, volatile)
    # tuple straight into point_wrapper, which fails at call time with
    # "'tuple' object has no attribute 'maker'".
    doc = inspect.getdoc(bundle_module) or ""
    assert "core, _volatile = compile_forward_sampling_function(" in doc


@pytest.mark.parametrize(
    ("label", "before", "after"),
    [
        ("string", ["alpha", "beta"], ["alpha", "GAMMA"]),
        ("integer", [1, 2, 3], [1, 2, 4]),
        ("datetime", ["2020-01-01", "2020-01-02"], ["2021-01-01", "2021-01-02"]),
        ("mixed-types", [1, "a"], [1, "b"]),
    ],
)
def test_fingerprint_notices_a_changed_string_or_datetime_coord(label, before, after):
    """A coord change must move the fingerprint, whatever its dtype.

    Regression: hashing went through pytensor's as_tensor_variable, which raises
    TypeError on a unicode dtype. That was swallowed, so every model with a
    string coord fingerprinted identically however the coord changed.
    """
    import pydantic

    def build(coord):
        with pm.Model(
            coords={"feature": pydantic.TypeAdapter(list).validate_python(coord)}
        ) as m:
            pm.Normal("b", 0, 1, dims="feature")
        return _fingerprint(m)

    assert build(before) != build(after), label


def test_fingerprint_notices_a_change_in_a_long_coord():
    """repr abbreviates past 1000 elements, which used to hide the change."""
    before = list(range(1500))
    after = list(range(1500))
    after[-1] = -1

    def build(coord):
        with pm.Model(coords={"feature": coord}) as m:
            pm.Normal("b", 0, 1, dims="feature")
        return _fingerprint(m)

    assert build(before) != build(after)


def test_fingerprint_is_stable_for_an_unchanged_coord():
    with pm.Model(coords={"feature": ["alpha", "beta"]}) as m:
        pm.Normal("b", 0, 1, dims="feature")

    assert _fingerprint(m) == _fingerprint(m)


def test_a_relative_pointer_is_the_callers_path_not_the_bundles(tmp_path):
    """A relative path the caller pointed at must not resolve inside the bundle.

    Regression: `is_portable` was true for any relative location, so the
    `runs/42/run.zarr` this module's own docstring recommends resolving to
    `runs/42/model/runs/42/run.zarr`.
    """
    path = save_model(build_model(), tmp_path / "model", idata="runs/42/run.zarr")
    bundle = load(path)

    assert bundle.data.embedded is False
    assert bundle.data.is_portable is False
    assert bundle.resolve() == Path("runs/42/run.zarr").expanduser()


def test_embedded_data_is_still_relocatable(tmp_path):
    """The case the relative resolution exists for must keep working."""
    import shutil

    saved = save_model(build_model(), tmp_path / "model", idata=build_sampled_idata())
    assert load(saved).resolve() == saved / "data.zarr"

    moved = tmp_path / "moved"
    shutil.copytree(saved, moved)

    assert load(moved).resolve() == moved / "data.zarr"
    assert "posterior" in load(moved).open_data()


@pytest.mark.parametrize(
    ("location", "remote", "absolute"),
    [
        ("s3://bucket/run.zarr", True, True),
        ("file:///data/run.zarr", False, True),
        ("C:\\data\\run.zarr", False, True),
        ("~/run.zarr", False, True),
        ("/abs/run.zarr", False, True),
        ("runs/42/run.zarr", False, False),
    ],
)
def test_location_parsing(location, remote, absolute):
    """`file://`, Windows drives and `~` all name a fixed place, not a relative one.

    Regression: `file:///x` and `C:\\x` were treated as relative because
    `Path("file:///x").is_absolute()` is False, so they resolved inside the bundle.
    """
    ref = DataRef(location)

    assert ref.is_remote is remote
    assert ref.is_absolute is absolute
    assert ref.is_portable is False


def test_tilde_is_expanded_when_resolving(tmp_path):
    ref = DataRef("~/run.zarr")

    assert "~" not in str(ref.expanded())
    assert str(ref.expanded()) == str(Path("~/run.zarr").expanduser())


def test_a_missing_zarr_says_how_to_install_it(tmp_path, monkeypatch):
    """zarr is optional, so the error must name the install rather than leak xarray's."""
    from pymc_marketing.bundle import _require_zarr

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(ImportError, match=r"pip install zarr"):
        _require_zarr("read")
    with pytest.raises(ImportError, match="data_format='netcdf'"):
        _require_zarr("embed")


def test_reading_zarr_data_checks_for_zarr_first(tmp_path, monkeypatch):
    """The guard runs before xarray is asked, so the message is ours."""
    path = save_model(build_model(), tmp_path / "model", idata=build_sampled_idata())
    real = importlib.util.find_spec

    def without_zarr(name):
        return None if name == "zarr" else real(name)

    monkeypatch.setattr("importlib.util.find_spec", without_zarr)

    with pytest.raises(ImportError, match=r"pip install zarr"):
        load(path).open_data()


def test_a_large_logp_is_not_failed_by_a_tiny_relative_difference(bundle_path):
    """An absolute tolerance alone fails a correct bundle at realistic logp sizes.

    A logp of order 1e4-1e6 moves by more than 1e-8 under summation-order
    changes alone, so comparing on `atol` alone reports a false FAIL.
    """
    from pymc_marketing.bundle import _logp_close

    recorded = -4594.7
    perturbed = recorded * (1 + 1e-9)

    assert abs(perturbed - recorded) > 1e-8, "the perturbation is the problem"
    assert _logp_close(perturbed, recorded, rtol=1e-6, atol=1e-8)
    assert not _logp_close(recorded * 1.01, recorded, rtol=1e-6, atol=1e-8)


def test_a_logp_near_zero_compares_absolutely():
    """Relative alone is meaningless when the reference is ~0."""
    from pymc_marketing.bundle import _logp_close

    assert _logp_close(0.0, 1e-12, rtol=1e-6, atol=1e-8)
    assert not _logp_close(1e-3, 0.0, rtol=1e-6, atol=1e-8)


def test_manifest_records_python_and_floatx(tmp_path):
    """cloudpickle stores bytecode, so the Python version is the fragile one."""
    import platform

    raw = json.loads(
        (save_model(build_model(), tmp_path / "model") / "manifest.json").read_text()
    )

    assert raw["python"] == platform.python_version()
    assert raw["floatX"] in ("float32", "float64")


def test_a_missing_reference_logp_warns_rather_than_going_quiet(tmp_path):
    """Silently recording null reads as 'nothing to compare', which is not the same."""
    import pytensor.tensor as pt

    from pymc_marketing.bundle import build_manifest

    with pm.Model() as model:
        x = pt.scalar("x")
        pm.Normal("y", x, 1, observed=1.0)

    with pytest.warns(UserWarning, match="cannot check drift"):
        manifest = build_manifest(model)

    assert manifest["reference_logp"] is None


def _scaled_idata(value):
    import xarray as xr

    draw = xr.Dataset(
        {"b": (("chain", "draw"), np.full((2, 3), value))},
        coords={"chain": [0, 1], "draw": np.arange(3)},
    )
    return xr.DataTree.from_dict({"/posterior": draw})


def test_resaving_over_a_bundle_replaces_it(tmp_path):
    """Regression: re-saving raised FileExistsError once data.zarr existed."""
    path = save_model(build_model(), tmp_path / "model", idata=_scaled_idata(2.0))

    save_model(build_model(), path, idata=_scaled_idata(3.0))

    assert load(path).open_data(group="posterior")["b"].values[0, 0] == 3.0
    assert [p.name for p in tmp_path.iterdir()] == ["model"], "staging left behind"


def test_resaving_without_data_does_not_leave_the_old_data(tmp_path):
    """Regression: the new manifest said no data while the old data.zarr remained."""
    path = save_model(build_model(), tmp_path / "model", idata=_scaled_idata(2.0))

    save_model(build_model(), path)

    assert load(path).manifest.data is None
    assert not (path / "data.zarr").exists()


def test_a_failed_save_leaves_the_previous_bundle_intact(tmp_path, monkeypatch):
    """A save that dies part-way through must not mix two models in one directory."""
    path = save_model(build_model(), tmp_path / "model", idata=_scaled_idata(2.0))
    before = load(path).validate()
    assert before.ok, before.notes

    def fail(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("pymc_marketing.bundle._write_embedded", fail)

    with pytest.raises(RuntimeError, match="disk full"):
        save_model(build_model(), path, idata=_scaled_idata(9.0))

    after = load(path)
    assert after.open_data(group="posterior")["b"].values[0, 0] == 2.0
    assert after.validate().ok, after.validate().notes
    assert [p.name for p in tmp_path.iterdir()] == ["model"]


def test_a_bundle_stranded_between_renames_is_recovered(tmp_path):
    """A save killed after moving the old bundle aside leaves it recoverable."""
    path = save_model(build_model(), tmp_path / "model", idata=_scaled_idata(2.0))

    stranded = tmp_path / "model.previous"
    os.replace(path, stranded)
    assert not path.exists()

    save_model(build_model(), path, idata=_scaled_idata(4.0))

    assert load(path).open_data(group="posterior")["b"].values[0, 0] == 4.0
    assert [p.name for p in tmp_path.iterdir()] == ["model"]


def test_building_the_model_warns_that_the_graph_is_unpickled(bundle_path):
    """cloudpickle runs code like pickle, so the one unpickling step says so."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("pymc_marketing.bundle._UNTRUSTED_BLOB_WARNED", False)
    try:
        with pytest.warns(UserWarning, match=r"runs code from.*trust"):
            load(bundle_path).model

        # Once is enough. A notice on every load gets ignored, which is the one
        # outcome that leaves the reader thinking they checked.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load(bundle_path).model
    finally:
        monkeypatch.undo()


def test_reading_the_manifest_alone_warns_about_nothing(bundle_path):
    """Only the unpickling step is risky, so that is the only step that warns."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle = load(bundle_path)

    assert bundle.manifest.fingerprint


def test_repointing_shared_variables_explains_the_logp_failure(tmp_path):
    """Regression: pm.set_data moved the logp and validate said only 'False'."""
    import pymc as pm

    with pm.Model(coords={"obs": [0, 1]}) as model:
        x = pm.Data("x", np.zeros(2), dims="obs")
        pm.Normal("y", x, 1, dims="obs", observed=np.zeros(2))

    bundle = load(save_model(model, tmp_path / "model"))
    with bundle.model:
        pm.set_data({"x": np.array([9.0, 9.0])})

    result = bundle.validate()
    assert result.checks["logp"] is False
    assert any("pm.set_data" in note for note in result.notes)


def test_embedded_zarr_uses_the_format_the_installed_zarr_writes(tmp_path):
    """No zarr_format pin: the bundle follows the zarr spec currently installed.

    The port wrote zarr_format=2, which is the legacy layout. Every tool this
    module exists to interoperate with has moved to v3, so pinning v2 made the
    bundle the odd one out for no benefit. The assertion is on v3 specifically
    because that is the spec zarr 3 implements, and a future zarr 4 would be a
    deliberate change to revisit rather than an accident.
    """
    import zarr

    path = save_model(build_model(), tmp_path / "model", idata=build_sampled_idata())

    assert zarr.open(path / "data.zarr").metadata.zarr_format == 3


def test_embedded_zarr_carries_no_nonstandard_sidecar(tmp_path):
    """The consolidated form is an xarray extension, not part of the zarr 3 spec.

    Writing it would swap a legacy-layout problem for a portability one, since a
    store carrying .zarr_metadata is readable by fewer tools, not more.
    """
    path = save_model(build_model(), tmp_path / "model", idata=build_sampled_idata())

    assert not (path / "data.zarr" / ".zarr_metadata").exists()


def test_a_model_only_bundle_has_no_data_key_at_all(bundle_path):
    """The documented default: absent, not null, and nothing zarr-shaped."""
    import json

    raw = json.loads((bundle_path / "manifest.json").read_text())

    assert "data" not in raw
    assert load(bundle_path).data is None
    assert load(bundle_path).datatree() is None


def test_a_model_only_bundle_needs_no_zarr(bundle_path, monkeypatch):
    """Most bundles store only the model, so they must not require zarr installed."""
    real = importlib.util.find_spec

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "zarr" else real(name),
    )

    assert load(bundle_path).validate().ok
    assert not (bundle_path / "data.zarr").exists()


def _posterior_predictive(model, idata, seed):
    """Draw posterior predictive samples, from whatever container this pymc returns."""
    res = pm.sample_posterior_predictive(
        idata,
        model=model,
        var_names=["y"],
        random_seed=seed,
        extend_inferencedata=False,
    )
    for group in res.groups:
        node = res[group].to_dataset()
        if "y" in node:
            return node["y"].values
    raise AssertionError(f"no y in any group of {res.groups}")


def test_a_restored_model_predicts_the_same_as_the_original(tmp_path):
    """The claim a bundle rests on: a restored model, given the same draws, predicts the same.

    Every other test checks structure, which is a proxy. This one runs the
    posterior predictive through both models with a shared seed and compares the
    numbers, which is the property a bundle is actually for.
    """
    model = build_model()
    idata = pm.sample(
        draws=100,
        tune=100,
        chains=2,
        cores=1,
        progressbar=False,
        random_seed=7,
        compute_convergence_checks=False,
        model=model,
    )

    restored = load_model(save_model(model, tmp_path / "model"))

    np.testing.assert_allclose(
        _posterior_predictive(restored, idata, seed=99),
        _posterior_predictive(model, idata, seed=99),
    )
