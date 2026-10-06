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
"""Save a PyMC Model to a bundle and get it back.

The model and the inference data are kept separate on purpose.  This module
persists the Model; you persist whatever ``pm.sample`` produced wherever you
already keep it, with plain xarray.  Combining the two is your call, and both
halves stay readable by anything that speaks zarr.

Writing the model
-----------------

::

    from pymc_marketing.bundle import audit, load, load_model, save_model

    report = audit(model)  # dry run, writes nothing
    if report.ok:
        path = save_model(model, "runs/42/model")  # -> PosixPath

The second argument is any local directory you like, written as a path so it
reads as one. Nothing here infers a bundle from its name.

Writing the inference data
--------------------------

``idata=`` is what ``pm.sample`` returned, not the model. It is a whole
``xarray.DataTree``: the posterior, but also ``sample_stats``, ``observed_data``,
and anything else your workflow put there. You own it. Keeping it separate is
one line and needs nothing from this package::

    idata.to_zarr("runs/42/run.zarr")  # every group
    idata["posterior"].to_zarr("runs/42/post.zarr")  # or one group of it

Or embed it in the bundle, keeping the groups you name. Anything a ``pm.sample``
or prior/posterior predictive workflow produces can go in: ``posterior``,
``posterior_predictive``, ``prior_predictive``, ``observed_data``,
``constant_data``, ``sample_stats``::

    save_model(model, "runs/42/model", idata=idata)  # all groups
    save_model(
        model,
        "runs/42/model",
        idata=idata,
        data_groups=("posterior", "posterior_predictive"),
    )  # just those

Embedded data is recorded *relative* to the bundle, so the directory can be
moved or copied and still find its own data.

Or pass neither and point at wherever you already keep it::

    save_model(model, "runs/42/model", idata=DataRef("s3://bucket/runs/42/run.zarr"))

Reading it back
---------------

::

    model = load_model("runs/42/model")  # a real pm.Model
    bundle = load("runs/42/model")  # Model plus metadata

    bundle.manifest.fingerprint  # structural hash, to detect drift
    bundle.manifest.version_mismatch  # recorded pins vs this environment
    bundle.validate()  # recompile logp and compare

    bundle.data.location  # where the data is, embedded or not
    bundle.open_data("posterior")  # xr.Dataset, for embedded or local data

Compiling a prediction function
-------------------------------

You often want the prediction function rather than the model, for serving. A
bundle stores the model, so you compile it once on load. ``pm.sample``'s
posterior predictive already does this through
:func:`pymc.sampling.compile_forward_sampling_function`, which is public, and
reusing it keeps your graph identical to the one you validated::

    from pymc.sampling import compile_forward_sampling_function
    from pymc.util import point_wrapper

    # it returns (function, volatile_rvs), so unpack before wrapping
    core, _volatile = compile_forward_sampling_function(
        outputs=[model["y"]],  # symbolic, not pm.draw(...)
        vars_in_trace=["a"],  # the names you supply per call
        basic_rvs=model.basic_RVs,
        model=model,
        random_seed=1,
    )
    predict = point_wrapper(core)
    predict(a=draws)

**Call it once at startup, not on the request path.** The first call is much
more expensive than the ones after it, and by how much depends on your backend
and whether its on-disk cache is warm. Two measurements of the same model and
code, one env with a cold numba cache and one with a warm one:

======================================== ============ ============
step                                     cold cache   warm cache
======================================== ============ ============
``from_parts`` + ``model_from_fgraph``   0.5 ms       0.5 ms
``compile_forward_sampling_function``    329 ms       7 ms
first predict call                       150 ms       3 ms
steady-state predict call                0.02 ms      0.02 ms
======================================== ============ ============

The steady-state cost is stable; the startup cost is not, and it shrinks as the
backend caches fill. Treat the top three rows as a budget to measure for your own
model and backend rather than numbers to rely on. The one that does not vary is
the last: every request after the first is a fraction of a millisecond.

What does *not* vary is that the cost is paid once, not per bundle. A second
model compiled in the same process reuses the warm caches, so a service holding
several bundles pays the startup cost once at boot rather than once per model.

You *can* pickle the compiled function instead, and it round-trips across
processes. It is still the wrong thing to persist here, for three reasons. It
ties the artifact to the pytensor that linked it rather than merely imported it.
Its pickled ``Generator`` is its own, so the draw sequence does not continue
from the model's. And you get back a callable, not a ``pm.Model``, so you cannot
refit or change the predictive. Persisting the model costs one recompile and
keeps all of that.

Saving somewhere else
---------------------

``save_model`` writes a local directory, which is the common case. It is not the
only route: the two steps are public, so you can write the same bundle anywhere
without giving up ``load`` or ``validate``.

Nothing here talks to a filesystem. ``save_model`` takes a local ``Path``, and
the compose route below hands you a ``dict`` and ``bytes``; where they go is
entirely yours. So this package depends on no filesystem library at all, not
fsspec and not zarr. The remote example uses an fsspec filesystem that *you*
create and that *you* would have to add to your own dependencies::

    import json

    import fsspec  # your dependency, not ours

    from pymc_marketing.bundle import ModelBundle, build_manifest, serialize_graph

    fs = fsspec.filesystem("s3")  # needs s3fs, also yours

    manifest = build_manifest(model, idata="s3://bucket/runs/42/run.zarr")
    blob = serialize_graph(model)

    fs.pipe_file("s3://bucket/runs/42/model/manifest.json", json.dumps(manifest))
    fs.pipe_file("s3://bucket/runs/42/model/graph.cloudpickle", blob)

    # and back, from wherever you keep it
    bundle = ModelBundle.from_parts(
        json.loads(fs.cat("s3://bucket/runs/42/model/manifest.json")),
        fs.cat("s3://bucket/runs/42/model/graph.cloudpickle"),
    )
    bundle.validate()

Any object with the methods you need will do, not just fsspec: a database
cursor, an HTTP client, a ``tarfile``. ``save_model`` is exactly the compose
route plus two local writes, so a bundle written either way is
indistinguishable. What stays fixed is the contract: the manifest keys, and the
two files. What you choose is the pickler, the destination, and any extra files
you want alongside.

What a bundle is
----------------

Two files, both plain: a JSON manifest and a cloudpickled graph::

    model/
      manifest.json       version pins, fingerprint, reference logp, metadata
      graph.cloudpickle   the model structure
      data.zarr           only if you embedded data, and only the groups named

``manifest.json`` and ``graph.cloudpickle`` are the whole contract and they are
fixed.  The **directory** is not: ``save_model`` creates whatever local path you
give it, so ``runs/42/model``, ``churn/`` and ``churn.model`` are all equally
valid, and nothing here infers a bundle from its name. A URL is refused, since
``Path("s3://b/runs/42/model")`` would otherwise create a directory named ``s3:``; use
the compose route below for anywhere but a local disk.
``data.zarr`` appears only when you embed, named by ``data_format``.

The graph is pymc's own round trip: ``pymc.model.fgraph.fgraph_from_model``
builds a ``pytensor.FunctionGraph`` and ``model_from_fgraph`` reverses it.  This
package owns neither, it only serializes the result and describes it.

Two consequences of that round trip, both tested:

* ``model_from_fgraph`` calls ``Model(model=None)``, so a **subclass** of
  ``pm.Model`` comes back as a plain ``Model``. Everything structural survives;
  the Python type does not. Wrapper classes that *contain* a Model are
  unaffected, because they hand you a plain ``pm.Model`` to begin with.
* ``model_from_fgraph`` dispatches with ``isinstance``, so a **subclass** of one
  of its marker Ops keeps the same branch and lands in the same model collection.

The manifest deliberately does *not* copy the model structure.  Names, roles,
dims, transforms and coords already live in the graph, as ``ModelVar`` props and
``fgraph._coords``, and come back with the reconstructed Model.  What the graph
cannot know is what wrote it, so that plus a fingerprint is all that is stored.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pymc as pm
    import xarray as xr

__all__ = [
    "AuditReport",
    "Blocker",
    "DataRef",
    "Manifest",
    "ModelBundle",
    "ValidationResult",
    "audit",
    "build_manifest",
    "load",
    "load_model",
    "save_model",
    "serialize_graph",
]

SCHEMA_VERSION = 1
#: Label for a bundle assembled from bytes rather than read off a directory.
_IN_MEMORY = "<memory>"
MANIFEST_FILE = "manifest.json"
GRAPH_FILE = "graph.cloudpickle"

# Zarr is not involved, but pymc declares cloudpickle (requirements.txt) and uses
# it for its own idata hashing, so serializing the graph adds no new dependency.
# stdlib pickle cannot resolve pymc's dynamically-created Op classes (CustomDist
# and friends), which is why this is not the default pickler.

#: Suffixes we recognise when inferring a data format from a location.
_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")


@dataclass(frozen=True)
class DataRef:
    """A pointer to inference data that lives outside the bundle.

    The bundle never writes or reads the referenced data. It only records where
    the caller put it, so that a consumer reading ``manifest.json`` can find the
    samples without knowing the naming convention in advance.

    The *name* is standardized; the *format* is not. ``location`` may be a
    ``.zarr`` directory, a ``.nc`` file, or an object-store URL, and ``format`` is
    only a hint for choosing an opener.

    Parameters
    ----------
    location : str
        Path or URL of the data artifact, exactly as the caller wrote it.
    format : str, optional
        ``"zarr"`` or ``"netcdf"``. Inferred from the suffix when omitted.
    groups : tuple of str, optional
        Which groups the artifact is expected to contain, for a consumer that
        wants to check before opening.

    Examples
    --------
    .. code-block:: python

        DataRef("runs/42/run.zarr", groups=("posterior",))
        DataRef("s3://bucket/runs/42/run.nc4", format="netcdf")

    Notes
    -----
    A ``location`` is never resolved by this package. Remote URLs are recorded
    and handed back unchanged, so no filesystem or cloud library is needed to
    read or write a manifest.
    """

    location: str
    format: str | None = None
    groups: tuple[str, ...] = ()

    @staticmethod
    def _infer_format(location: str) -> str:
        return "netcdf" if location.lower().endswith(_NETCDF_SUFFIXES) else "zarr"

    def __post_init__(self):
        """Normalize location, infer format, and freeze groups to a tuple."""
        # frozen dataclass, so normalize through object.__setattr__
        object.__setattr__(self, "location", str(self.location))
        if not self.format:
            object.__setattr__(self, "format", self._infer_format(self.location))
        object.__setattr__(self, "groups", tuple(self.groups))

    @classmethod
    def coerce(cls, value: DataRef | str | Path) -> DataRef:
        """Accept a DataRef, or a bare path whose suffix implies the format.

        Raises
        ------
        TypeError
            For anything else, rather than quietly turning it into a location.
        """
        if isinstance(value, DataRef):
            return value
        if not isinstance(value, str | Path):
            raise TypeError(
                f"data must be a DataRef, a path, or an xarray object, got {type(value).__name__}"
            )
        return cls(str(value))

    def as_dict(self) -> dict:
        """JSON-serializable form, with a fixed set of keys."""
        return {
            "location": self.location,
            "format": self.format,
            "groups": list(self.groups),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DataRef:
        """Rebuild from a manifest entry written by :meth:`as_dict`."""
        return cls(
            location=raw["location"],
            format=raw.get("format"),
            groups=tuple(raw.get("groups", ())),
        )

    @property
    def is_remote(self) -> bool:
        """True for a URL with a scheme, e.g. ``s3://bucket/run.zarr``."""
        return "://" in self.location and not self.location.startswith("file://")

    @property
    def is_portable(self) -> bool:
        """True when the location is relative and so travels with the bundle.

        ``save_model`` records embedded data relative to the bundle directory, so
        a bundle can be moved or copied and still find its own data.
        """
        return not self.is_remote and not Path(self.location).is_absolute()

    def exists(self) -> bool:
        """Whether the location resolves. Remote URLs are not probed."""
        if self.is_remote:
            return True
        return Path(self.location).expanduser().exists()


@dataclass(frozen=True)
class Manifest:
    """The durable half of a bundle: plain JSON, readable without pymc.

    Deliberately does not hold the model structure. Names, roles, dims,
    transforms and coords are already in the graph and are re-derivable from
    the reconstructed Model. What the graph cannot know is what wrote it, and
    that is what this records.
    """

    raw: dict

    def __getitem__(self, key):
        """Return a manifest key, so a Manifest reads like the dict it wraps."""
        return self.raw[key]

    def get(self, key, default=None):
        """Return a manifest key, or ``default`` when it is absent."""
        return self.raw.get(key, default)

    def __contains__(self, key) -> bool:
        """Report whether the wrapped manifest has this key."""
        return key in self.raw

    def __repr__(self) -> str:
        """Return a representation naming the schema version and creation time."""
        return f"Manifest(schema_version={self['schema_version']}, created={self.get('created')!r})"

    @property
    def fingerprint(self) -> str | None:
        """Structural hash, so drift is detectable without storing the structure."""
        return self.raw.get("fingerprint")

    @property
    def reference_logp(self) -> float | None:
        """Logp at the initial point, as recorded at save time."""
        value = self.raw.get("reference_logp")
        return float(value) if value is not None else None

    @property
    def metadata(self) -> dict:
        """Return the free-form metadata the caller stored, verbatim."""
        return dict(self.raw.get("metadata", {}))

    @property
    def data(self) -> DataRef | None:
        """Where the inference data lives, or None if the bundle makes no claim."""
        raw = self.raw.get("data")
        return DataRef.from_dict(raw) if raw else None

    @property
    def created(self) -> str | None:
        """Return when the bundle was written, as recorded in the manifest."""
        return self.raw.get("created")

    def check_schema(self) -> None:
        """Raise if the manifest is newer than this package understands.

        An older manifest is fine: fields are only ever added, so an unknown
        reader can pass one along untouched.
        """
        found = self.raw.get("schema_version", 1)
        if found > SCHEMA_VERSION:
            raise ValueError(
                f"manifest declares schema_version={found}, but this "
                f"pymc_marketing understands {SCHEMA_VERSION}. Upgrade pymc-marketing, "
                f"or read the manifest yourself with ModelBundle.from_parts and "
                f"keep the graph."
            )

    def version_mismatch(self) -> dict[str, tuple[str, str]]:
        """Return the recorded pins against the live environment.

        Informational, not fatal, but worth surfacing: a cloudpickled graph is
        not guaranteed to load under a different pymc or pytensor.
        """
        live = {
            "pymc": _version("pymc"),
            "pytensor": _version("pytensor"),
            "pymc_marketing": _version("pymc_marketing"),
        }
        return {
            k: (self.raw[k], live[k])
            for k in live
            if k in self.raw and self.raw[k] != live[k]
        }


def _transform_label(transform: Any) -> str:
    """Return a stable name for a transform, without depending on its class location.

    ``type(t).__name__`` breaks if a transform is renamed or moved between
    modules; ``t.name`` is the public string pymc reports elsewhere.
    """
    if transform is None:
        return "none"
    return getattr(transform, "name", None) or type(transform).__name__


def _fingerprint(model: pm.Model) -> str:
    """Structural hash covering what ``model_from_fgraph`` should preserve.

    Order is excluded on purpose: it is not preserved and does not matter.
    Dims are joined to a string so the hash does not care whether they arrive as
    a tuple or a list.

    Only comparable within one pytensor version: coord values go through
    ``Constant.signature``, whose semantics pytensor owns. A drift there shows up
    as a "structure changed" note from :meth:`ModelBundle.validate`, never as a
    failed check.
    """
    import pytensor.tensor as pt

    h = hashlib.sha256()
    h.update(("named:" + ",".join(sorted(model.named_vars))).encode())
    for collection, label in (
        ("free_RVs", "free"),
        ("observed_RVs", "obs"),
        ("deterministics", "det"),
        ("potentials", "pot"),
        ("data_vars", "data"),
    ):
        h.update(
            f"|{label}:{','.join(sorted(_names(getattr(model, collection))))}".encode()
        )
    h.update(("|vv:" + ",".join(sorted(_names(model.value_vars)))).encode())
    for name in sorted(model.named_vars):
        h.update(
            f"|dim:{name}={'/'.join(model.named_vars_to_dims.get(name) or ())}".encode()
        )
    transforms = model.rvs_to_transforms or {}
    for key in sorted(transforms, key=str):
        name = key if isinstance(key, str) else key.name
        h.update(f"|tr:{name}={_transform_label(transforms[key])}".encode())
    coords = getattr(model, "_coords", None) or getattr(model, "coords", None) or {}
    for key in sorted(coords):
        try:
            # pytensor's own value signature: stable across processes, unlike a
            # repr, and unlike len() it notices the values and not just length
            # signature() is on Variable at runtime; the stubs do not say so.
            sig = pt.as_tensor_variable(coords[key]).signature()  # type: ignore[attr-defined]
        except Exception:  # noqa: S112 - a coord we cannot put on a tensor
            # contributes nothing to the hash, and skipping it keeps an exotic
            # coord from making every model look changed.
            continue
        h.update(
            f"|coord:{key}={hashlib.sha256(repr(sig).encode()).hexdigest()[:16]}".encode()
        )
    return h.hexdigest()[:32]


@dataclass(frozen=True)
class Blocker:
    """Something that will stop a Model round-tripping, and what to do."""

    kind: str
    detail: str
    remedy: str


@dataclass
class AuditReport:
    """The result of :func:`audit`."""

    blockers: list[Blocker] = field(default_factory=list)
    #: Whether ``pickle`` alone cannot round-trip this graph in *this* process.
    #: That is not a portability guarantee: a class defined in ``__main__``
    #: pickles by reference, so this can be False for a graph that still needs
    #: its classes present elsewhere.
    stdlib_pickle_fails: bool = False
    n_nodes: int = 0
    depth: int = 0
    #: The graph that was audited, so callers can reuse it instead of rebuilding.
    fgraph: Any = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """Return True when no blocker applies."""
        return not self.blockers

    def kinds(self) -> set[str]:
        """Return the distinct blocker kinds, for pattern matching in tests."""
        return {b.kind for b in self.blockers}

    def __str__(self) -> str:
        """Return the report as lines: sizes, then any blocker and its remedy."""
        lines = [f"nodes={self.n_nodes} depth={self.depth} ok={self.ok}"]
        if self.stdlib_pickle_fails:
            lines.append("  ! stdlib pickle will not work; cloudpickle is required")
        lines.extend(
            f"  x {b.kind}: {b.detail}\n      remedy: {b.remedy}" for b in self.blockers
        )
        return "\n".join(lines)


def _fgraph_or_blocker(model: pm.Model) -> tuple[Any, Blocker | None]:
    """Build the model's graph, or report why it cannot be built.

    Shared so that auditing, describing and serializing a model all work from one
    build. ``fgraph_from_model`` is not free, and three separate calls mean the
    manifest can describe a different graph than the bytes written next to it.
    """
    from pymc.model.fgraph import fgraph_from_model

    try:
        fgraph, _ = fgraph_from_model(model)
    except NotImplementedError as exc:
        return None, Blocker("fgraph", str(exc), "see the initial_values blocker")
    except ValueError as exc:
        return None, Blocker("fgraph", str(exc), "export the top-level model")
    return fgraph, None


def audit(model: pm.Model, *, fgraph: Any = None) -> AuditReport:
    """Check whether a Model can be serialized, without writing anything.

    Parameters
    ----------
    model : pm.Model
        The model to inspect.
    fgraph : pytensor.graph.fg.FunctionGraph, optional
        An already built graph, to avoid a second build. The report carries the
        one it audited as ``report.fgraph``.

    Returns
    -------
    AuditReport
        ``ok`` is False if any blocker applies.

    Raises
    ------
    TypeError
        If ``model`` is not a ``pymc.Model``. This is a programming error rather
        than a model problem, so it is not a blocker. In practice it means a
        wrapper object was passed where its ``.model`` was meant: ``MMM``,
        ``LinearModel``, ``BayesianSARIMAX`` and every other ``ModelBuilder``
        *contain* a Model rather than inheriting from it.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.bundle import audit

        report = audit(model)
        if not report.ok:
            for blocker in report.blockers:
                print(blocker.kind, blocker.detail)
    """
    import pymc

    if not isinstance(model, pymc.Model):
        raise TypeError(
            f"expected a pymc.Model, got {type(model).__name__}. If this wraps a "
            f"model rather than being one, pass its .model attribute instead."
        )

    report = AuditReport()

    if any(v is not None for v in (model.rvs_to_initial_values or {}).values()):
        report.blockers.append(
            Blocker(
                "initial_values",
                "model sets initvals=, which fgraph_from_model does not represent",
                "clear them before export, or add initval to ModelValuedVar.__props__",
            )
        )
    if getattr(model, "parent", None) is not None:
        report.blockers.append(
            Blocker(
                "submodel",
                "model is nested under a parent",
                "export the top-level model",
            )
        )

    if fgraph is None:
        fgraph, blocker = _fgraph_or_blocker(model)
        if blocker is not None:
            report.blockers.append(blocker)
            return report

    report.fgraph = fgraph
    report.n_nodes, report.depth = _stats(fgraph)
    report.stdlib_pickle_fails = _stdlib_pickle_fails(fgraph)
    report.blockers.extend(_customdist_blockers(model))
    return report


def _stdlib_pickle_fails(fgraph) -> bool:
    """Report whether the standard library cannot round-trip this graph.

    Only ever called on a graph we just built in this process, so the ``loads``
    here executes our own output rather than anything untrusted.
    """
    import pickle

    try:
        pickle.loads(pickle.dumps(fgraph, -1))  # noqa: S301 - our own graph
    except Exception:
        return True
    return False


def _customdist_blockers(model: pm.Model) -> list[Blocker]:
    """CustomDist(random=...) without an explicit logp= cannot reload.

    Such a distribution derives logp numerically and registers it against the
    dynamically-created Op class. Serializing by value produces a *different*
    class, so the registration no longer applies and compile_logp raises.

    A symbolic ``dist=`` is fine: it builds a pure pytensor graph up front, and
    its Op has no ``_random_fn`` at all. Both forms name their class
    ``CustomDist_<var>``, so the presence of ``_random_fn`` is the only
    discriminator.

    Free and observed RVs are both inspected: a free CustomDist is what
    ``pm.sample`` trips over, and it is just as unserializable.
    """
    found = []
    for rv in list(model.free_RVs) + list(model.observed_RVs):
        if not rv.owner:
            continue
        op_type = type(rv.owner.op)
        if "CustomDist" not in op_type.__name__:
            continue
        random_fn = getattr(op_type, "_random_fn", None)
        if random_fn is None:
            continue
        if getattr(op_type, "_logprob_fn", None) in (None, random_fn):
            found.append(
                Blocker(
                    "customdist_random",
                    f"{rv.name}: CustomDist(random=...) without an explicit logp= "
                    "loses its logp once serialized",
                    "pass logp= explicitly, or use a symbolic dist=",
                )
            )
    return found


def serialize_graph(model: pm.Model, *, fgraph: Any = None) -> bytes:
    """Serialize the model's graph, without deciding where it goes.

    Use this together with :func:`build_manifest` when you need to write a bundle
    somewhere :func:`save_model` does not cover: an object store, a registry, a
    container image, a database row. The result is the same bytes
    :func:`save_model` would write.

    Parameters
    ----------
    model : pm.Model
        The model to serialize.
    fgraph : pytensor.graph.fg.FunctionGraph, optional
        An already built graph, to avoid a second build.

    Returns
    -------
    bytes
        A cloudpickled ``pytensor.FunctionGraph``. cloudpickle rather than
        stdlib pickle because pymc builds Op classes dynamically and they cannot
        be resolved by import path.

    Examples
    --------
    .. code-block:: python

        blob = serialize_graph(model)
        write_somewhere("model/graph.cloudpickle", blob)

    ``write_somewhere`` is yours; this function only produces bytes.
    """
    import cloudpickle

    if fgraph is None:
        fgraph, blocker = _fgraph_or_blocker(model)
        if blocker is not None:
            raise ValueError(f"cannot serialize this model:\n  {blocker.detail}")
    return cloudpickle.dumps(fgraph, protocol=-1)


#: File name for data embedded in a bundle, by format.
EMBEDDED_NAMES = {"zarr": "data.zarr", "netcdf": "data.nc"}
#: Formats we can write and read, and the opener each needs.
_FORMATS = ("zarr", "netcdf")


def _looks_remote(value: str | Path) -> bool:
    """Return True for anything carrying a URL scheme, so we never mkdir a ``s3:`` dir."""
    return "://" in str(value)


def _netcdf_engine(action: str) -> Literal["h5netcdf", "netcdf4"]:
    """Pick a netCDF engine that can handle groups, or say why none can.

    A DataTree is a group, and xarray's default netCDF engine is scipy, which
    cannot write or read groups. Both h5netcdf and netCDF4 can, so require one of
    them rather than letting the failure surface from inside xarray.
    """
    import importlib.util

    engines: tuple[tuple[Literal["h5netcdf", "netcdf4"], str], ...] = (
        ("h5netcdf", "h5netcdf"),
        ("netcdf4", "netCDF4"),
    )
    for engine, module in engines:
        if importlib.util.find_spec(module) is not None:
            return engine
    raise ImportError(
        f"cannot {action} netCDF data: it needs h5netcdf or netCDF4, because the "
        f"bundle stores groups and xarray's default netCDF engine (scipy) cannot "
        f"handle them. Install one, or use data_format='zarr'."
    )


def _write_embedded(idata, dest: Path, fmt: str, groups) -> None:
    """Write a Dataset or DataTree into ``dest`` using plain xarray.

    ``groups`` selects which groups to keep when ``idata`` is a DataTree; an empty
    selection keeps all of them, so a full ``pm.sample`` result can be stored
    whole.
    """
    import xarray as xr

    if isinstance(idata, xr.DataTree):
        names = tuple(groups) if groups else tuple(idata.keys())
        missing = [g for g in names if g not in idata.keys()]
        if missing:
            raise KeyError(
                f"idata has no group(s) {missing}; available: {sorted(idata.keys())}"
            )
        tree = xr.DataTree.from_dict({f"/{g}": idata[g].to_dataset() for g in names})
    elif isinstance(idata, xr.Dataset):
        name = groups[0] if groups else "data"
        tree = xr.DataTree.from_dict({f"/{name}": idata})
    else:
        raise TypeError(
            f"cannot embed {type(idata).__name__}; pass a DataRef or a path instead"
        )

    if fmt == "netcdf":
        # str() because xarray's to_netcdf overloads do not accept a Path here.
        tree.to_netcdf(str(dest), engine=_netcdf_engine("write"))
        return

    try:
        tree.to_zarr(dest, zarr_format=2)
    except ImportError as exc:
        # xarray says "Missing optional dependency 'zarr'", which is true and
        # unhelpful: say what to do instead.
        raise ImportError(
            "cannot embed data as zarr, because zarr is not installed. "
            "Install zarr, or pass data_format='netcdf', which needs "
            "h5netcdf or netCDF4."
        ) from exc


def _groups_of(idata) -> tuple[str, ...]:
    """Group names of a DataTree or Dataset, without leading slashes."""
    import xarray as xr

    if isinstance(idata, xr.DataTree):
        return tuple(idata.keys())
    return ("data",)


def build_manifest(
    model: pm.Model,
    *,
    metadata: dict | None = None,
    idata: DataRef | str | Path | xr.Dataset | xr.DataTree | None = None,
    data_groups: tuple[str, ...] | None = None,
    data_format: str = "zarr",
    fgraph: Any = None,
) -> dict:
    """Describe the model as a plain dict, without deciding where it goes.

    Pair this with :func:`serialize_graph` to write a bundle by any route you
    like. The keys are the bundle's contract, so a bundle written this way is
    indistinguishable from one :func:`save_model` produced.

    Parameters
    ----------
    model : pm.Model
        The model to describe.
    metadata : dict, optional
        Free-form, must be JSON-serializable. Anything else raises, naming the
        key, rather than being stringified behind your back.
    idata : DataRef, path, Dataset or DataTree, optional
        Either a pointer to inference data stored elsewhere, or a Dataset or
        DataTree to embed. When embedding, the location is written by
        :func:`save_model`, not here.
    data_groups : tuple of str, optional
        Which groups to embed. Empty or omitted keeps all of them, so a whole
        ``pm.sample`` result can be stored. Only meaningful when embedding.
    fgraph : pytensor.graph.fg.FunctionGraph, optional
        An already built graph, to avoid a second build.

    Returns
    -------
    dict
        JSON-serializable manifest.
    """
    if fgraph is None:
        fgraph, blocker = _fgraph_or_blocker(model)
        if blocker is not None:
            raise ValueError(f"cannot describe this model:\n  {blocker.detail}")
    n_nodes, depth = _stats(fgraph)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        # the graph cannot record what wrote it
        "pymc": _version("pymc"),
        "pytensor": _version("pytensor"),
        "pymc_marketing": _version("pymc_marketing"),
        "n_nodes": n_nodes,
        "depth": depth,
        # change detection rather than a structural copy
        "fingerprint": _fingerprint(model),
        # one float, so validation needs no data store
        "reference_logp": _reference_logp(model),
        "metadata": dict(metadata or {}),
    }
    if idata is not None:
        manifest["data"] = _data_ref(
            idata, data_groups, embedded=True, data_format=data_format
        ).as_dict()
    return manifest


def _data_ref(idata, groups, *, embedded: bool, data_format: str = "zarr") -> DataRef:
    """Describe ``idata`` for the manifest, whether embedded or pointed at."""
    import xarray as xr

    if isinstance(idata, xr.Dataset | xr.DataTree):
        if not embedded:
            raise TypeError(
                "a DataTree or Dataset can only be embedded by save_model; "
                "use build_manifest with a DataRef or path if you are writing the files yourself"
            )
        kept = tuple(groups) if groups else _groups_of(idata)
        return DataRef(EMBEDDED_NAMES[data_format], format=data_format, groups=kept)
    return DataRef.coerce(idata)


def save_model(
    model: pm.Model,
    path: str | Path,
    *,
    metadata: dict | None = None,
    idata: DataRef | str | Path | xr.Dataset | xr.DataTree | None = None,
    data_groups: tuple[str, ...] | None = None,
    data_format: str = "zarr",
    strict: bool = True,
) -> Path:
    """Serialize ``model`` into a bundle directory and return the path.

    This is the convenience route: it writes ``manifest.json`` and
    ``graph.cloudpickle`` into a local directory. For anywhere else, compose
    :func:`build_manifest` and :func:`serialize_graph` yourself and write them
    however you like; a bundle built that way loads through :func:`load` or
    :meth:`ModelBundle.from_parts` unchanged.

    Parameters
    ----------
    model : pm.Model
        The model to save. Must not set ``initvals=`` and must not be nested.
    path : str or Path
        Directory to create.
    metadata : dict, optional
        Free-form, must be JSON-serializable. Stored verbatim.
    idata : DataRef, path, Dataset or DataTree, optional
        Where the inference data is. Pass a ``DataRef`` or a path to record a pointer to
        data you already wrote, or pass the ``DataTree`` / ``Dataset`` itself to
        embed it in the bundle. Anything in a ``pm.sample`` result can be stored:
        posterior, posterior_predictive, observed_data, constant_data,
        prior_predictive, sample_stats.
    data_groups : tuple of str, optional
        Which groups to embed. Omit to keep all of them. Only used when
        embedding.
    data_format : {"zarr", "netcdf"}
        Format the data will be embedded as. Ignored when ``data`` is a pointer.
    strict : bool
        When True (default) refuse to save a model that :func:`audit` flags.

    Returns
    -------
    Path
        The bundle directory.

    Raises
    ------
    ValueError
        If ``strict`` and ``audit(model)`` reports a blocker.

    Examples
    --------
    .. code-block:: python

        path = save_model(model, "runs/42/model", metadata={"tier": 3})
    """
    report = audit(model)
    if strict and report.blockers:
        raise ValueError(
            "cannot save this model:\n  "
            + "\n  ".join(
                f"{b.kind}: {b.detail}\n      remedy: {b.remedy}"
                for b in report.blockers
            )
        )

    import xarray as xr

    if data_format not in _FORMATS:
        raise ValueError(f"data_format must be one of {_FORMATS}, got {data_format!r}")
    embedding = isinstance(idata, xr.Dataset | xr.DataTree)

    manifest = build_manifest(
        model,
        metadata=metadata,
        idata=idata,
        data_groups=data_groups,
        data_format=data_format,
        fgraph=report.fgraph,
    )
    blob = serialize_graph(model, fgraph=report.fgraph)
    # Render before creating anything, so a bad manifest leaves no half-built
    # directory behind.
    text = _dump_json(manifest)

    if _looks_remote(path):
        raise ValueError(
            f"cannot write a bundle to {path!r}: save_model only writes to a local "
            f"directory, and given a URL it would silently create a directory named "
            f"{str(path).split('/')[0]!r} instead. Use build_manifest and "
            f"serialize_graph, then write the two parts wherever you like, and read "
            f"them back with ModelBundle.from_parts."
        )
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_FILE).write_text(text)
    (root / GRAPH_FILE).write_bytes(blob)

    if embedding:
        _write_embedded(
            idata, root / EMBEDDED_NAMES[data_format], data_format, data_groups or ()
        )
    return root


def _reference_logp(model: pm.Model) -> float | None:
    """Logp at the initial point, so :meth:`ModelBundle.validate` has something to compare."""
    try:
        import numpy as np

        point = model.initial_point()
        value = float(np.asarray(model.compile_logp()(point), dtype=float))
    except Exception:
        return None
    # NaN and inf are not JSON, and a reference that cannot be compared is worse
    # than no reference at all.
    return value if np.isfinite(value) else None


def _unserializable(value: Any, path: str = "manifest"):
    """Yield the paths of values ``json`` cannot encode, for error messages."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _unserializable(item, f"{path}[{key!r}]")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _unserializable(item, f"{path}[{index}]")
    else:
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            yield f"{path} ({type(value).__name__})"


def _dump_json(manifest: dict) -> str:
    """Serialize the manifest, naming the metadata that is not JSON-native.

    ``default=str`` would let the write succeed and hand back a string where the
    caller passed an array or a timestamp, which is a corruption nobody notices
    until it is load-bearing.
    """
    try:
        return json.dumps(manifest, indent=2, allow_nan=False)
    except (TypeError, ValueError) as err:
        offenders = ", ".join(_unserializable(manifest))
        raise type(err)(
            f"{err}. Not JSON-serializable at: {offenders}. Convert it yourself, or pass a string."
        ) from err


class ModelBundle:
    """A loaded bundle. The Model itself is built lazily on first access."""

    def __init__(self, path: str | Path):
        root = Path(path)
        if not (root / MANIFEST_FILE).exists():
            raise FileNotFoundError(
                f"{root} does not look like a bundle (no {MANIFEST_FILE})"
            )
        manifest = json.loads((root / MANIFEST_FILE).read_text())
        wrapped = Manifest(manifest)
        wrapped.check_schema()
        if not (root / GRAPH_FILE).exists():
            raise FileNotFoundError(
                f"{root} has a {MANIFEST_FILE} but no {GRAPH_FILE}, so the model "
                f"is missing. The bundle was probably copied incompletely, or the "
                f"graph was written somewhere else on purpose; in the latter case "
                f"use ModelBundle.from_parts."
            )
        self._blob = (root / GRAPH_FILE).read_bytes()
        self.path = root
        self.manifest = wrapped
        self._model: pm.Model | None = None

    @classmethod
    def from_parts(
        cls,
        manifest: dict | str | bytes,
        graph: bytes,
        *,
        path: str | Path = _IN_MEMORY,
    ) -> ModelBundle:
        """Build a bundle from a manifest and a graph you fetched yourself.

        The counterpart to :func:`build_manifest` and :func:`serialize_graph`: use
        this when the bundle did not come from a local directory, such as an
        object store, a registry, or a database.

        Parameters
        ----------
        manifest : dict or str or bytes
            The manifest, as a dict or as raw JSON.
        graph : bytes
            The serialized graph from :func:`serialize_graph`.
        path : str or Path, optional
            Directory the parts were fetched into. Defaults to
            ``"<memory>"``, which is only a label for ``repr`` and error
            messages. Pass the real directory whenever the manifest records
            embedded data, because those locations are relative to the bundle
            and :meth:`resolve` needs somewhere to resolve them against.

        Returns
        -------
        ModelBundle

        Examples
        --------
        .. code-block:: python

            # a pointer to data you already have works with no directory
            bundle = ModelBundle.from_parts(
                read_somewhere("model/manifest.json"),
                read_somewhere("model/graph.cloudpickle"),
            )

            # embedded data needs the directory it was downloaded into
            bundle = ModelBundle.from_parts(
                manifest_dict,
                graph_bytes,
                path=downloaded_dir,
            )

        ``read_somewhere`` is yours, and may be a string or bytes; both are accepted.
        """
        raw = json.loads(manifest) if not isinstance(manifest, dict) else manifest
        wrapped = Manifest(raw)
        wrapped.check_schema()
        bundle = cls.__new__(cls)
        bundle.path = Path(path)
        bundle.manifest = wrapped
        bundle._blob = graph
        bundle._model = None
        return bundle

    @property
    def data(self) -> DataRef | None:
        """The recorded data pointer, or None if the bundle makes no claim."""
        return self.manifest.data

    def resolve(self, ref: DataRef | None = None) -> str | Path:
        """Resolve a pointer to something openable.

        A relative location is taken as relative to the bundle, which is how
        embedded data is recorded, so a bundle can be moved and still find its
        own data. Absolute paths and remote URLs are returned unchanged.
        """
        ref = ref if ref is not None else self.data
        if ref is None:
            raise ValueError(f"{self.path} records no data pointer")
        if not ref.is_portable:
            return ref.location
        if str(self.path) == _IN_MEMORY:
            # Embedded data travels with the bundle, so a bundle assembled from
            # bytes in memory has nowhere to look. MLflow and friends hand the
            # bytes back alongside the directory they were fetched into; pass
            # that directory as ``path`` so relative pointers still resolve.
            raise ValueError(
                f"this bundle records data at {ref.location!r}, which is relative, "
                f"but it was built from memory rather than read from a directory. "
                f"Pass the directory the parts came from: "
                f"ModelBundle.from_parts(manifest, graph, path=that_directory)."
            )
        return self.path / ref.location

    def open_data(self, group: str | None = None):
        """Open the referenced or embedded data with xarray.

        Works for both a pointer and data embedded by ``save_model``.

        Parameters
        ----------
        group : str, optional
            A group name such as ``"posterior"``. Omit it to get the whole tree.

        Returns
        -------
        xarray.Dataset or xarray.DataTree

        Raises
        ------
        ValueError
            If the bundle records no data pointer, or its format is not one we
            know how to open.
        """
        import xarray as xr

        ref = self.data
        if ref is None:
            raise ValueError(f"{self.path} records no data pointer")
        if ref.format not in _FORMATS:
            raise ValueError(
                f"cannot open data of format {ref.format!r}; this package knows "
                f"{list(_FORMATS)}. Record format='zarr' or format='netcdf' in the "
                f"manifest to fix it, or open it yourself; the bundle never has to read it."
            )

        target = self.resolve(ref)
        if ref.format == "netcdf":
            tree = xr.open_datatree(target, engine=_netcdf_engine("read"))
        else:
            tree = xr.open_datatree(target, engine="zarr")
        return tree if group is None else tree[group].to_dataset()

    def datatree(self):
        """Return the stored data as a whole DataTree, or None if there is none."""
        return None if self.data is None else self.open_data()

    @property
    def model(self) -> pm.Model:
        """The reconstructed Model.

        Dims round trip as-is. pymc used to hand ``model_from_fgraph`` a list
        where ``add_named_variable`` declares a tuple, which made a cloned
        model's ``named_vars_to_dims`` compare unequal to the original's; that
        is fixed upstream, and nothing here works around it.
        """
        if self._model is None:
            import cloudpickle
            from pymc.model.fgraph import model_from_fgraph

            self._model = model_from_fgraph(cloudpickle.loads(self._blob))
        return self._model

    def validate(self, atol: float = 1e-8) -> ValidationResult:
        """Check this bundle really is the Model that was saved.

        Returns
        -------
        ValidationResult
            ``ok`` reflects only the load-bearing checks. Version drift,
            structural drift and variable reordering are reported as notes,
            because none of them by itself means the bundle is wrong.

            ``ok`` is False when no check could be performed at all, so a
            ``PASS`` always means something was actually verified.
        """
        result = ValidationResult(path=self.path)

        result.version_mismatch = self.manifest.version_mismatch()
        if result.version_mismatch:
            result.notes.append(
                "version drift: "
                + ", ".join(
                    f"{k} {a} -> {b}" for k, (a, b) in result.version_mismatch.items()
                )
            )

        current = _fingerprint(self.model)
        if current != self.manifest.fingerprint:
            result.notes.append(
                f"structure changed: {self.manifest.fingerprint} -> {current}"
            )

        recorded = self.manifest.reference_logp
        try:
            import numpy as np

            actual = float(
                np.asarray(self.model.compile_logp()(self.model.initial_point()))
            )
            result.logp = actual
        except Exception as exc:
            # A model that cannot compute logp is unusable, whether or not a
            # reference was recorded. Reporting this as a failure is the point:
            # a bundle that verifies nothing must not print PASS.
            result.checks["logp"] = False
            result.differences.append(f"logp: {type(exc).__name__}: {exc}")
        else:
            if recorded is None:
                # Nothing to compare against, which is not the same as broken.
                # Still checked, so that ok=True always means something was.
                result.checks["logp_computable"] = True
                result.notes.append("no reference_logp recorded; nothing to compare")
            else:
                result.reference_logp = recorded
                result.logp_delta = abs(actual - recorded)
                result.checks["logp"] = result.logp_delta <= atol
                if not result.checks["logp"]:
                    result.differences.append(
                        f"logp drifted: recorded {recorded!r}, now {actual!r}"
                    )

        result.ok = (
            bool(result.checks)
            and all(result.checks.values())
            and not result.differences
        )
        return result

    def __repr__(self) -> str:
        """Return a representation naming the bundle path and its data pointer."""
        ref = self.data
        if ref is None:
            return f"ModelBundle({str(self.path)!r})"
        return f"ModelBundle({str(self.path)!r}, data={ref.location!r})"


@dataclass
class ValidationResult:
    """Outcome of :meth:`ModelBundle.validate`."""

    path: Path
    ok: bool = True
    checks: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    version_mismatch: dict = field(default_factory=dict)
    logp: float | None = None
    reference_logp: float | None = None
    logp_delta: float | None = None

    def __str__(self) -> str:
        """Return the result as lines: verdict, each check, then differences and notes."""
        lines = [f"{'PASS' if self.ok else 'FAIL'}  {self.path}"]
        lines.extend(f"  {'ok ' if v else 'X  '} {k}" for k, v in self.checks.items())
        if self.logp_delta is not None:
            lines.append(f"  logp {self.logp!r} vs recorded {self.reference_logp!r}")
            lines.append(f"  max|delta| = {self.logp_delta:.3g}")
        lines.extend(f"  x  {d}" for d in self.differences)
        lines.extend(f"  ~  {n}" for n in self.notes)
        return "\n".join(lines)


def load(path: str | Path) -> ModelBundle:
    """Open a bundle without rebuilding the Model.

    Parameters
    ----------
    path : str or Path
        A directory written by :func:`save_model`.

    Returns
    -------
    ModelBundle
    """
    return ModelBundle(path)


def load_model(path: str | Path) -> pm.Model:
    """Return the Model. The short path: hand me a bundle, get a Model.

    Parameters
    ----------
    path : str or Path
        A directory written by :func:`save_model`.

    Returns
    -------
    pm.Model

    Examples
    --------
    .. code-block:: python

        model = load_model("runs/42/model")
    """
    return ModelBundle(path).model


def _stats(fgraph) -> tuple[int, int]:
    """Node count and graph depth."""
    memo: dict = {}

    def depth(variable) -> int:
        if variable in memo:
            return memo[variable]
        memo[variable] = 0
        memo[variable] = (
            1 + max((depth(i) for i in variable.owner.inputs), default=0)
            if variable.owner
            else 0
        )
        return memo[variable]

    return len(fgraph.apply_nodes), max((depth(o) for o in fgraph.outputs), default=0)


def _names(sequence) -> list[str]:
    """Model collections mix treelists of Variables with treedicts keyed by name."""
    return [v if isinstance(v, str) else v.name for v in sequence]


def _version(module: str) -> str:
    """Return ``module.__version__``, for the drift pins in the manifest.

    Deliberately not wrapped in a bare ``except``: the three callers are all hard
    dependencies, so a failure here means the environment is broken rather than
    the version being unknown. Swallowing it wrote ``"unknown"`` into the
    manifest, and because :meth:`Manifest.version_mismatch` compared
    ``"unknown"`` against ``"unknown"`` it never reported the drift the pin
    exists to warn about.
    """
    import importlib

    return importlib.import_module(module).__version__
