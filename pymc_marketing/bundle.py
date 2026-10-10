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

.. code-block:: python

    from pymc_marketing.bundle import audit, load, load_model, save_model

    report = audit(model)  # dry run, writes nothing
    if report.ok:
        path = save_model(model, "runs/42/model")  # -> PosixPath

The second argument is any local directory you like, written as a path so it
reads as one. Nothing here infers a bundle from its name.

``idata`` is optional, and leaving it out is the common case: a model on its
own is two files, ``manifest.json`` and ``model.cloudpickle``, with no ``data``
key in the manifest at all, rather than an empty or null pointer. The next
section covers what changes when you pass one.

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

.. code-block:: python

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
and whether its on-disk cache is warm. Unpickling and compiling are cheap next
to that first predict: it is the call that fills the cache, and it is seconds
cold against milliseconds warm, while every request after it is a fraction of a
millisecond. The startup cost moves with the backend; the steady-state cost does
not. Measure the startup cost for your own model rather than trusting anyone's
numbers, including these.

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

    from pymc_marketing.bundle import ModelBundle, build_manifest, serialize_model

    fs = fsspec.filesystem("s3")  # needs s3fs, also yours

    manifest = build_manifest(model, idata="s3://bucket/runs/42/run.zarr")
    blob = serialize_model(model)

    fs.pipe_file("s3://bucket/runs/42/model/manifest.json", json.dumps(manifest))
    fs.pipe_file("s3://bucket/runs/42/model/model.cloudpickle", blob)

    # and back, from wherever you keep it
    bundle = ModelBundle.from_parts(
        json.loads(fs.cat("s3://bucket/runs/42/model/manifest.json")),
        fs.cat("s3://bucket/runs/42/model/model.cloudpickle"),
    )
    bundle.validate()

Any object with the methods you need will do, not just fsspec: a database
cursor, an HTTP client, a ``tarfile``. ``save_model`` is that compose route
plus the writes: the two files, and the data file when you hand it a Dataset or
a DataTree. The two routes agree on the manifest provided the compose route,
which writes nothing, is handed a pointer and not an object. What stays fixed
is the contract: the manifest keys, and the two files. What you choose is the
pickler, the destination, and any extra files you want alongside.

What a bundle is
----------------

A JSON manifest and a cloudpickled Model, plus data only if you embedded it:

.. code-block:: text

    model/
      manifest.json       version pins, fingerprint, reference logp, metadata
      model.cloudpickle   the Model itself, pickled
      data.zarr           only if you embedded data, and only the groups named

``manifest.json`` and ``model.cloudpickle`` are the whole contract and they are
fixed, as are the keys inside the manifest.  The **directory** is not: ``save_model``
creates whatever local path you
give it, so ``runs/42/model``, ``churn/`` and ``churn.model`` are all equally
valid, and nothing here infers a bundle from its name. A URL is refused, since
``Path("s3://b/runs/42/model")`` would otherwise create a directory named ``s3:``; use
the compose route below for anywhere but a local disk.
``data.zarr`` appears only when you embed, named by ``data_format``.

The Model itself is what gets serialized: cloudpickled whole, and unpickled on
the way back.  Not lowered to a ``pytensor.FunctionGraph`` and rebuilt with
``pymc.model.fgraph.model_from_fgraph``.  Both routes round trip the same numbers,
but rebuilding re-derives ``named_vars_to_dims`` and lands a list where the Model
holds a tuple, so dims do not survive on a pymc that has not fixed it
(pymc-devs/pymc#8465).  ``Model.copy()`` and ``copy.deepcopy`` go through the same
pair and have the same defect, so the Model is pickled rather than rebuilt.

That also means a **subclass** of ``pm.Model`` comes back as its own type, with
its state, and a class defined only in a notebook comes back by value without
needing that code at load time.  Wrapper classes that *contain* a Model are
unaffected, because they hand you a plain ``pm.Model`` to begin with.

For an **Op** the same by-value rebuild is the failure mode rather than the
feature.  pymc registers logprob implementations on the class object, so a
rebuilt class has none and ``compile_logp`` raises.  That is the one thing
:func:`audit` refuses; a Model subclass and a notebook-defined Op are handled
oppositely despite being pickled the same way.

**Scope: a raw** ``pm.Model`` **in the environment that wrote it.**  This is a
same-environment artefact, not an archival format: it depends on the pymc,
pytensor and Python that produced the model, which is why the manifest pins
them.  For a builder class such as ``MMM``, use ``ModelIO.save`` / ``ModelIO.load``,
which calls your class again, stores your configuration, and survives you editing
the class.  This module stores the Model as built, which is the faithful answer
for a model built without a builder and the least portable.  If a format that
crosses versions is what you need, this is not it.

The manifest deliberately does *not* copy the model structure.  Names, roles,
dims, transforms and coords travel with the Model, so re-deriving them into the
manifest would be a second thing to keep in step, and not the thing that is
actually loaded.  What a Model cannot know is what wrote it, so that plus a
fingerprint is all that is stored.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse, urlunparse

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
    "serialize_model",
]

SCHEMA_VERSION = 1

#: ``initial_point`` draws when an RV has ``initval="prior"``, so an unseeded
#: reference moves on every save and validate() fails a perfectly good bundle.
#: Both the writer and the checker must use this value or neither agrees.
_REFERENCE_SEED = 0
#: Label for a bundle assembled from bytes rather than read off a directory.
_IN_MEMORY = "<memory>"
MANIFEST_FILE = "manifest.json"
MODEL_FILE = "model.cloudpickle"

# Zarr is not involved, but pymc declares cloudpickle (requirements.txt) and uses
# it for its own idata hashing, so serializing the graph adds no new dependency.
# stdlib pickle cannot resolve pymc's dynamically-created Op classes (CustomDist
# and friends), which is why this is not the default pickler.

#: Suffixes we recognise when inferring a data format from a location.
_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf", ".netcdf")
_ZARR_SUFFIXES = (".zarr",)

#: A Windows drive-letter path, e.g. ``C:\\data``. ``Path("C:\\data").drive`` is
#: empty on POSIX, because pathlib does not treat backslash as a separator, so the
#: pattern has to be matched against the raw string.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class DataRef:
    """A pointer to inference data that lives outside the bundle.

    The bundle never writes or reads the referenced data. It only records where
    the caller put it, so that a consumer reading ``manifest.json`` can find the
    samples without knowing the naming convention in advance.

    The *name* is standardized; the *format* is not. ``location`` may be a
    ``.zarr`` directory, a ``.nc`` file, or an object-store URL, and ``format`` is
    only a hint for choosing an opener.

    **How a consumer must resolve ``location``.** The keys are fixed, and the
    rule for reading them is part of the contract:

    * ``embedded`` is ``true`` only for data this bundle wrote into itself, and
      its ``location`` is relative to the bundle directory. Resolve it against the
      bundle and the bundle is relocatable.
    * ``embedded`` is ``false`` for a path or URL the caller supplied. Resolve it
      as written, against the process working directory for a relative path. Do
      **not** resolve it against the bundle: a caller who wrote
      ``runs/42/run.zarr`` means that path, and joining it to the bundle
      directory would look for ``runs/42/model/runs/42/run.zarr``.
    * ``file://`` URLs and Windows drive letters name one fixed location and are
      never relative. ``~`` expands in the reading process.

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
    #: True only for data this bundle wrote into itself. A relative path that the
    #: caller pointed at is *not* portable, even though it looks relative.
    embedded: bool = False

    @staticmethod
    def _infer_format(location: str) -> str:
        """Read the format off the suffix, or refuse instead of guessing.

        This used to default everything unrecognised to zarr, so ``run.h5`` and
        ``run.parquet`` were recorded as zarr. The reader then reached for the
        wrong library and failed somewhere later, in a traceback that named
        neither this line nor the real mistake.
        """
        lowered = location.lower()
        if lowered.endswith(_NETCDF_SUFFIXES):
            return "netcdf"
        if lowered.endswith(_ZARR_SUFFIXES):
            return "zarr"
        raise ValueError(
            f"cannot tell the format of {location!r} from its name. Pass "
            "format='zarr' or format='netcdf' to say which it is."
        )

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
            An xarray object arrives here too, with a note, because the one
            place that can embed one is :func:`save_model`, which writes it.
        """
        if isinstance(value, DataRef):
            return value
        if isinstance(value, str | Path):
            return cls(str(value))
        raise TypeError(
            f"data must be a DataRef or a path, got {type(value).__name__}. A "
            "Dataset or DataTree is embedded by save_model, which is what "
            "writes it; this records where data already is."
        )

    def as_dict(self) -> dict:
        """JSON-serializable form, with a fixed set of keys."""
        return {
            "location": self.location,
            "format": self.format,
            "groups": list(self.groups),
            "embedded": self.embedded,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DataRef:
        """Rebuild from a manifest entry written by :meth:`as_dict`."""
        return cls(
            location=raw["location"],
            format=raw.get("format"),
            groups=tuple(raw.get("groups", ())),
            embedded=raw.get("embedded", False),
        )

    @property
    def is_remote(self) -> bool:
        """True for a URL this package cannot open itself, e.g. ``s3://bucket/run.zarr``.

        ``file://`` is deliberately *not* remote: it is a local path, and treating
        it as remote would hide the fact that ``Path("file:///x").is_absolute()``
        is False, which would otherwise make it look like a portable ref.
        """
        scheme = urlparse(self.location).scheme
        if scheme in ("", "file"):
            return False
        if _WINDOWS_DRIVE.match(self.location):
            return False
        return True

    @property
    def is_absolute(self) -> bool:
        """True for a location that names the same place in every process."""
        if self.is_remote:
            return True
        if urlparse(self.location).scheme == "file":
            return True
        # Windows drive letters are absolute on Windows, which is where such a
        # path would have been written, so treat them as absolute everywhere.
        if _WINDOWS_DRIVE.match(self.location):
            return True
        return Path(self.location).expanduser().is_absolute()

    @property
    def is_portable(self) -> bool:
        """True when the location travels with the bundle.

        Only data this bundle wrote into itself, and only when its location is
        relative. A *relative path the caller pointed at* is not portable: it is
        the caller's path, and resolving it against the bundle would look for
        ``runs/42/model/runs/42/run.zarr``.
        """
        return self.embedded and not self.is_remote and not self.is_absolute

    def expanded(self) -> Path:
        """Return the local path with ``~`` expanded."""
        text = self.location
        if urlparse(text).scheme == "file":
            text = urlunparse(urlparse(text)._replace(scheme=""))
        return Path(text).expanduser()

    def exists(self) -> bool:
        """Whether the location resolves. Remote URLs are not probed."""
        if self.is_remote:
            return True
        return self.expanded().exists()


@dataclass(frozen=True)
class Manifest:
    """The durable half of a bundle: plain JSON, readable without pymc.

    Deliberately does not hold the model structure. Names, roles, dims,
    transforms and coords travel with the Model, so copying them here would be a
    second thing to keep in step and not the thing that is actually loaded. What
    a Model cannot know is what wrote it, and that is what this records.

    **The key set is a contract.** A consumer of a bundle written by a compatible
    ``schema_version`` may rely on these keys being present with these meanings,
    and should ignore any key it does not recognise so that a newer writer is not
    a failure.

    ================== ==================================================
    key                what a consumer may rely on
    ================== ==================================================
    ``schema_version`` present, and greater than this package understands
                       means the manifest must not be loaded.
    ``created``        ISO 8601 UTC, second resolution. Informational.
    ``pymc``           version that wrote it. Drift is reported, never fatal.
    ``pytensor``       as above.
    ``pymc_marketing`` as above.
    ``python``         as above. The pin most likely to matter on load,
                       since cloudpickle stores classes as bytecode.
    ``floatX``         precision the reference logp was taken at.
    ``n_nodes``        graph size, informational. 0 when the graph could not
                       be built, which is not a failure.
    ``depth``          as above.
    ``fingerprint``    structural hash of names, roles, dims, transforms and
                       coord values. Change means the model changed.
    ``reference_logp`` logp at the initial point when saved. May be ``null``
                       when it could not be computed, which is warned at
                       save time rather than hidden.
    ``metadata``       your own JSON, verbatim.
    ``data``           a :class:`DataRef` under a fixed key set. **Absent**
                       when the bundle holds only the Model. Absent is not
                       the same as null.
    ================== ==================================================

    Adding a key is a minor change; removing one, renaming one, or changing what
    one means requires a ``schema_version`` bump. Fields have only ever been
    added, which is what lets an older reader pass a newer manifest through
    untouched rather than refusing it.
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

        Informational, not fatal, but worth surfacing: a pickled Model is
        not guaranteed to load under a different pymc or pytensor.
        """
        live = {
            "pymc": _version("pymc"),
            "pytensor": _version("pytensor"),
            "pymc_marketing": _version("pymc_marketing"),
            # cloudpickle stores dynamic classes as bytecode, so the Python that
            # writes a bundle is the version most likely to break loading it.
            "python": _python_version(),
            # The logp reference is only meaningful at the precision it was taken
            # at, so record the precision it was taken at.
            "floatX": _floatX(),
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
    """Structural hash of everything a bundle should hand back unchanged.

    Order is excluded on purpose: it is not preserved and does not matter.
    Dims are joined to a string so the hash does not care whether they arrive as
    a tuple or a list. Coord values are hashed from their dtype and bytes, so a
    string or datetime coord moves the hash when it changes, and a long coord is
    not abbreviated.
    """
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
        h.update(f"|coord:{key}={_coord_digest(coords[key])}".encode())
    return h.hexdigest()[:32]


def _coord_digest(value: Any) -> str:
    """Return a hash of a coord's dtype and exact bytes.

    Going through ``pytensor.tensor.as_tensor_variable`` looks tidier, but it
    rejects the dtypes that matter most here: ``["a", "b"]`` raises
    ``TypeError: Unsupported dtype for TensorType: <U1``. Swallowing that would
    fingerprint every model with a string coord, a datetime coord, or a pandas
    index the same however its coord changed, so the value is hashed directly.

    Hashing dtype, shape and bytes covers every dtype numpy can hold, and unlike
    ``repr`` it does not abbreviate long arrays to ``...``.
    """
    import numpy as np

    try:
        arr = np.asarray(value)
    except Exception:
        # Not array-like at all. Hash a repr rather than dropping the coord, so
        # that a change to it still moves the fingerprint.
        return "repr:" + hashlib.sha256(repr(value).encode()).hexdigest()[:16]
    if arr.dtype == object:
        # Object arrays hold arbitrary Python. Hash element reprs, which still
        # distinguishes values, rather than bailing out of the coord.
        try:
            joined = "\x00".join(repr(item) for item in arr.ravel().tolist())
        except Exception:
            joined = repr(value)
        return "object:" + hashlib.sha256(joined.encode()).hexdigest()[:16]
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(str(arr.shape).encode())
    digest.update(np.ascontiguousarray(arr).tobytes())
    return f"{arr.dtype}:{digest.hexdigest()[:16]}"


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
    #: Things worth saying that do not stop the save. A blocker demoted because
    #: the model turned out to round-trip anyway lands here.
    notes: list[str] = field(default_factory=list)
    n_nodes: int = 0
    depth: int = 0
    #: The graph that was audited, so callers can reuse it instead of rebuilding.
    #: None when the graph could not be built, which costs n_nodes and depth but
    #: not the bundle, since the Model is what actually gets written.
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
        lines.extend(f"  - {note}" for note in self.notes)
        lines.extend(
            f"  x {b.kind}: {b.detail}\n      remedy: {b.remedy}" for b in self.blockers
        )
        return "\n".join(lines)


def _fgraph_or_reason(model: pm.Model) -> tuple[Any, str | None]:
    """Build the model's graph, or say why it cannot be built.

    Shared so that auditing and describing a model agree on its size, since
    ``fgraph_from_model`` is not free and two builds could differ. The graph is
    informational: the Model is what gets written, so failing to build one costs
    ``n_nodes`` and ``depth`` rather than the bundle, and the caller records the
    reason as a note. A whole ``Blocker`` was built here before, with a remedy
    that no one could ever be shown, because this is never a refusal.
    """
    from pymc.model.fgraph import fgraph_from_model

    try:
        fgraph, _ = fgraph_from_model(model)
    except NotImplementedError as exc:
        return None, f"fgraph: {exc}"
    except ValueError as exc:
        return None, f"fgraph: {exc}"
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
        ``ok`` is False if any blocker applies. There is currently one kind: an
        Op whose logprob pymc registered on the class object itself, where that
        class cannot be imported again. Serializing rebuilds the class by value
        without the registration, so the bundle reloads into a model whose
        ``compile_logp`` raises ``Logprob method not implemented``. In practice
        that is a ``CustomDist`` given an explicit ``logp=``, whether or not it
        also has ``random=`` or a symbolic ``dist=``, and any Op defined where
        it cannot be imported again, such as a notebook. A symbolic ``dist=`` on
        its own is fine: its logp comes from the graph, so nothing is registered
        against the class.

        Things that used to block, and now only note, because the Model is
        pickled rather than rebuilt from a graph: a model that sets ``initval``,
        which travels with the Model; a nested model, which is pickled together
        with its parent and so makes a larger bundle; and a graph that cannot be
        built, which costs ``n_nodes`` and ``depth`` rather than the bundle.
        Their notes are worth reading, but none of them stops a save.

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

    if getattr(model, "parent", None) is not None:
        # A nested model does not own copies of its variables, it shares them with
        # the parent, so pickling it serialises the parent too. The bundle is valid
        # and loads correctly; it is larger, and it carries whatever data the parent
        # holds, which is worth saying but is not a reason to refuse the save.
        report.notes.append(
            "this model is nested, and a nested model is pickled together with the "
            "parent that owns it, so the bundle also carries the parent's variables "
            "and data; export the top-level model if you do not want that"
        )

    if fgraph is None:
        fgraph, reason = _fgraph_or_reason(model)
        if reason is not None:
            # The graph is no longer what gets written, so failing to build one
            # costs n_nodes and depth rather than the bundle. Note it and carry on.
            report.notes.append(reason)

    report.fgraph = fgraph
    if fgraph is not None:
        report.n_nodes, report.depth = _stats(fgraph)
    report.blockers.extend(_unserializable_logp_blockers(model))
    return report


def _pickled_by_reference(cls: type) -> bool:
    """Whether cloudpickle will store this class by reference rather than by value.

    This mirrors cloudpickle's own test (``_lookup_module_and_qualname`` in
    ``cloudpickle.cloudpickle``): by reference only when ``cls.__module__`` is not
    ``__main__``, that module is already imported, and reading ``cls.__qualname__``
    off it returns ``cls``. Everything else, including an interactive session or a
    notebook, is rebuilt by value in the process that loads the bundle.
    """
    import sys

    module_name = getattr(cls, "__module__", None)
    if module_name is None or module_name == "__main__":
        return False
    module = sys.modules.get(module_name)
    if module is None:
        return False
    try:
        return getattr(module, cls.__qualname__) is cls
    except AttributeError:
        return False


def _unserializable_logp_blockers(model: pm.Model) -> list[Blocker]:
    """Random variables that reload into a model that cannot evaluate their logp.

    pymc registers a logprob implementation against the Op class object itself,
    with ``_logprob.register(rv_type)`` in ``pymc.distributions.custom``, and those
    classes are made with ``type()``. Such a class is not reachable by import path,
    so cloudpickle rebuilds it by value, the registration does not travel with it,
    and ``compile_logp`` raises ``Logprob method not implemented`` in the process
    that loads the bundle.

    Both halves are needed to fail: the registration must be there, *and* the class
    must be unreachable. A symbolic ``dist=`` with no explicit ``logp=`` sets
    ``inline_logprob`` instead, so nothing is registered and it reloads correctly
    even though its class is just as unreachable. In the other direction, a class
    defined at the top level of an importable module is re-imported when the bundle
    is loaded, which re-runs the registration, so it is safe too.

    Free and observed RVs are both inspected: a free one is what ``pm.sample``
    trips over, and it is just as unserializable.
    """
    from pymc.logprob.abstract import _logprob

    found = []
    for rv in list(model.free_RVs) + list(model.observed_RVs):
        if not rv.owner:
            continue
        cls = type(rv.owner.op)
        if cls in _logprob.registry and not _pickled_by_reference(cls):
            found.append(
                Blocker(
                    "logp_registration_lost",
                    f"{rv.name}: pymc registered this Op's logprob on the class "
                    f"{cls.__name__}, and {cls.__module__!r} cannot supply that "
                    "class again, so loading would build a different class with "
                    "no logprob and compile_logp would raise",
                    "use a symbolic dist= and leave logp= unset, so the logp is "
                    "derived from the graph instead of registered against the class",
                )
            )
    return found


def serialize_model(model: pm.Model) -> bytes:
    """Serialize the model, without deciding where it goes.

    Use this together with :func:`build_manifest` when you need to write a bundle
    somewhere :func:`save_model` does not cover: an object store, a registry, a
    container image, a database row. The result is the same bytes
    :func:`save_model` would write.

    Parameters
    ----------
    model : pm.Model
        The model to serialize.

    Returns
    -------
    bytes
        A cloudpickled ``pm.Model``. cloudpickle rather than stdlib pickle
        because pymc builds Op classes dynamically and they cannot be resolved
        by import path.

    Notes
    -----
    The Model is pickled whole, rather than lowered to a
    ``pytensor.FunctionGraph`` and rebuilt with ``model_from_fgraph``. Both round
    trip the same numbers, but rebuilding re-derives ``named_vars_to_dims``
    through ``add_named_variable`` and lands a list where the Model holds a
    tuple, so dims do not survive. That is pymc#8465, and it costs a caller on a
    released pymc a Model whose declared dims disagree with its own.
    ``Model.copy()`` and ``copy.deepcopy`` go through the same pair and have the
    same defect, so pickling the Model avoids inheriting it.

    Examples
    --------
    .. code-block:: python

        blob = serialize_model(model)
        write_somewhere("model/model.cloudpickle", blob)

    ``write_somewhere`` is yours; this function only produces bytes.
    """
    import cloudpickle

    return cloudpickle.dumps(model, protocol=-1)


#: File name for data embedded in a bundle, by format.
EMBEDDED_NAMES = {"zarr": "data.zarr", "netcdf": "data.nc"}
#: Formats we can write and read, and the opener each needs.
_FORMATS = ("zarr", "netcdf")


def _looks_remote(value: str | Path) -> bool:
    """Return True for anything carrying a URL scheme, so we never mkdir a ``s3:`` dir."""
    return "://" in str(value)


def _logp_close(actual: float, recorded: float, rtol: float, atol: float) -> bool:
    """Compare two logp values on both a relative and an absolute scale.

    Relative alone would be wrong for a logp near zero, and absolute alone is
    hopeless for a logp of 1e6, where a relative difference of 1e-9 is already
    1e-3 in absolute terms.
    """
    import numpy as np

    return bool(np.isclose(actual, recorded, rtol=rtol, atol=atol))


def _require_zarr(action: str) -> None:
    """Raise with the install line if zarr is missing.

    zarr is not a declared dependency, so a clean install does not have it and the
    default ``data_format`` will not work until it is installed. That is a
    deliberate trade: a bundle that stores only the Model never reaches this, and
    a dependency nobody uses should not be forced on everyone. The error names
    the fix, which xarray's own does not.

    The alternative format is not free either, so say so rather than offering it
    as a way out: h5netcdf and netCDF4 are undeclared too.
    """
    import importlib.util

    if importlib.util.find_spec("zarr") is None:
        raise ImportError(
            f"cannot {action} zarr data: zarr is not installed, and pymc-marketing "
            f"does not declare it. Install it with `pip install zarr` or "
            f"`uv add zarr`. The other format, data_format='netcdf', needs h5netcdf "
            f"or netCDF4, which are not declared either."
        )


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

    _require_zarr("embed")
    # No zarr_format: take xarray's default, which follows the zarr spec the
    # installed zarr implements. Pinning 2 here wrote the legacy layout, which
    # the tools this module exists to interoperate with have moved away from.
    #
    # consolidated=False for the same reason. The consolidated form is a
    # .zarr_metadata sidecar that is not part of the zarr 3 specification, so
    # writing it would trade one portability problem for another. zarr also
    # warns about it on every write, and a reader cannot open the store faster
    # if the sidecar is missing anyway.
    tree.to_zarr(dest, consolidated=False)


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
    idata: DataRef | str | Path | None = None,
    fgraph: Any = None,
) -> dict:
    """Describe the model as a plain dict, without deciding where it goes.

    Pair this with :func:`serialize_model` to write a bundle by any route you
    like. The keys are the bundle's contract, so a bundle written this way is
    indistinguishable from one :func:`save_model` produced.

    Parameters
    ----------
    model : pm.Model
        The model to describe.
    metadata : dict, optional
        Free-form, must be JSON-serializable. Anything else raises, naming the
        key, rather than being stringified behind your back.
    idata : DataRef or path, optional
        Where inference data lives. This records the location and writes
        nothing. A Dataset or DataTree is refused rather than accepted, because
        a manifest claiming embedded data that nobody wrote is a bundle that
        fails when it is read. Use :func:`save_model` for that; it writes the
        file.
    fgraph : pytensor.graph.fg.FunctionGraph, optional
        An already built graph, to avoid a second build.

    Returns
    -------
    dict
        JSON-serializable manifest.
    """
    if fgraph is None:
        fgraph, _ = _fgraph_or_reason(model)
    # A graph we could not build costs n_nodes and depth, not the bundle: the
    # Model is what gets written, and it pickles either way.
    n_nodes, depth = _stats(fgraph) if fgraph is not None else (0, 0)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        # the graph cannot record what wrote it
        "pymc": _version("pymc"),
        "pytensor": _version("pytensor"),
        "pymc_marketing": _version("pymc_marketing"),
        "python": _python_version(),
        "floatX": _floatX(),
        "n_nodes": n_nodes,
        "depth": depth,
        # change detection rather than a structural copy
        "fingerprint": _fingerprint(model),
        # one float, so validation needs no data store
        "reference_logp": _reference_logp(model),
        "metadata": dict(metadata or {}),
    }
    if idata is not None:
        manifest["data"] = DataRef.coerce(idata).as_dict()
    return manifest


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
    ``model.cloudpickle`` into a local directory. For anywhere else, compose
    :func:`build_manifest` and :func:`serialize_model` yourself and write them
    however you like; a bundle built that way loads through :func:`load` or
    :meth:`ModelBundle.from_parts` unchanged.

    Parameters
    ----------
    model : pm.Model
        The model to save. Must be a ``pm.Model`` and not a class that wraps one;
        pass the wrapped ``.model`` instead. A nested model saves, and carries its
        parent with it, which :func:`audit` notes.
    path : str or Path
        Directory to create. It must not exist, be empty, or be a bundle written
        here before. A directory holding anything else is left untouched and
        refused, because replacing one is not recoverable.
    metadata : dict, optional
        Free-form, must be JSON-serializable. Stored verbatim.
    idata : DataRef, path, Dataset or DataTree, optional
        Where the inference data is. Omit to store the Model alone: the bundle is
        then just ``manifest.json`` and ``model.cloudpickle``, the manifest has no
        ``data`` key, and nothing zarr-related is imported. Note that the Model's
        own ``pm.Data`` values and ``observed`` arrays travel inside the graph
        either way, so a bundle without ``idata`` still holds the training data.

        Pass a ``DataRef`` or a path to record a pointer to
        data you already wrote, or pass the ``DataTree`` / ``Dataset`` itself to
        embed it in the bundle. Anything in a ``pm.sample`` result can be stored:
        posterior, posterior_predictive, observed_data, constant_data,
        prior_predictive, sample_stats.

        This is inference output only. The values the Model itself was built
        from, meaning ``pm.Data`` containers and the ``observed`` arrays, are part
        of the Model and always go into ``model.cloudpickle`` whether or not you
        pass ``idata``. So a bundle holds your training data, including anything
        personal in it, and should be treated as sensitive wherever it is stored.
        Replace those values with empty containers before saving if the bundle
        has to leave a trusted boundary.

        A relative path you pass here is recorded exactly as written and resolved
        by the caller, against the process working directory. It is not resolved
        against the bundle, so the same bundle reads the same data only from the
        same working directory. Pass an absolute path, or embed the data, when the
        bundle has to be readable from elsewhere.
    data_groups : tuple of str, optional
        Which groups to embed. Omit to keep all of them. Only used when
        embedding.
    data_format : {"zarr", "netcdf"}
        Format the data will be embedded as. Ignored when ``idata`` is a pointer.
    strict : bool
        When True (default) refuse to save a model that :func:`audit` flags.

    Returns
    -------
    Path
        The bundle directory.

    Notes
    -----
    Saving builds the bundle in a sibling directory and moves it into place at the
    end, so an interrupted save cannot leave a mixture of two models. Saving over
    an existing bundle replaces it wholesale, which means data the previous
    bundle held is not carried over: passing no ``idata`` to a re-save drops the
    old ``data.zarr`` rather than leaving it stranded next to a manifest that no
    longer mentions it.

    Overwriting is limited to an empty directory, or one holding exactly the files
    this module writes. A directory with anything else in it, a report, a log, a
    ``.git``, is somebody's data and is refused rather than emptied for them.

    Raises
    ------
    ValueError
        If ``strict`` and ``audit(model)`` reports a blocker, or if ``path``
        already exists and is not empty and not a bundle.

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

    # Writing the file is this function's job, so embedding is decided here and
    # the manifest is handed a finished DataRef rather than the object itself.
    if isinstance(idata, xr.Dataset | xr.DataTree):
        ref = DataRef(
            EMBEDDED_NAMES[data_format],
            format=data_format,
            groups=tuple(data_groups) if data_groups else _groups_of(idata),
            embedded=True,
        )
    elif idata is None:
        ref = None
    elif isinstance(idata, DataRef | str | Path):
        ref = DataRef.coerce(idata)
    else:
        raise TypeError(
            "idata must be a DataRef, a path, a Dataset, or a DataTree, got "
            f"{type(idata).__name__}"
        )

    manifest = build_manifest(
        model,
        metadata=metadata,
        idata=ref,
        fgraph=report.fgraph,
    )
    blob = serialize_model(model)
    # Render before creating anything, so a bad manifest leaves no half-built
    # directory behind.
    text = _dump_json(manifest)

    if _looks_remote(path):
        raise ValueError(
            f"cannot write a bundle to {path!r}: save_model only writes to a local "
            f"directory, and given a URL it would silently create a directory named "
            f"{str(path).split('/')[0]!r} instead. Use build_manifest and "
            f"serialize_model, then write the two parts wherever you like, and read "
            f"them back with ModelBundle.from_parts."
        )
    # Build in a sibling and move it into place, so the destination is either the
    # old bundle or the new one and never a mixture. Writing directly meant a
    # failure part-way through left a manifest describing one model beside the
    # previous model's data, and a re-save over an existing data.zarr raised
    # FileExistsError *after* the manifest and graph had already been replaced.
    # ``Path(".").name`` is "", so the sibling names have to come from a path
    # that has one. ``absolute`` and not ``resolve``: on macOS resolve rewrites
    # /var to /private/var, and the directory we then write is no longer the
    # one asked for.
    root = Path(path).absolute()
    staging = root.with_name(root.name + ".incomplete")
    backup = root.with_name(root.name + ".previous")
    _check_destination_is_a_bundle(root)
    _recover_interrupted_save(root, backup)
    _clear_staging(staging)

    try:
        staging.mkdir(parents=True)
        (staging / MANIFEST_FILE).write_text(text)
        (staging / MODEL_FILE).write_bytes(blob)
        if ref is not None and ref.embedded:
            _write_embedded(
                idata,
                staging / EMBEDDED_NAMES[data_format],
                data_format,
                data_groups or (),
            )
        # Inside the try, so a swap that fails does not leave a half-built
        # staging directory behind for the next save to trip over.
        _swap_into_place(staging, root, backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return root


_UNTRUSTED_BLOB_WARNED = False


def _warn_untrusted_blob() -> None:
    """Say plainly that reading the graph runs code from whoever wrote it.

    cloudpickle is a pickle: unpickling can import modules and call anything the
    payload names. pickle has no equivalent of ``load`` versus a restricted
    reader, so the honest options are to trust the bundle or not read it at all,
    and silence here is how people lose track of which one they did.

    Once per process. The step happens once per bundle, but ``load_model`` builds
    a fresh bundle per call, and a warning that fires on every ordinary load
    trains people to ignore it.
    """
    global _UNTRUSTED_BLOB_WARNED
    if _UNTRUSTED_BLOB_WARNED:
        return

    import warnings

    warnings.warn(
        "loading a bundle unpickles model.cloudpickle, which runs code from "
        "whoever wrote the bundle, the same way pickle.load does. Only load a "
        "bundle you produced yourself or otherwise trust.",
        UserWarning,
        stacklevel=3,
    )
    _UNTRUSTED_BLOB_WARNED = True


_BUNDLE_ENTRIES = frozenset({MANIFEST_FILE, MODEL_FILE, *EMBEDDED_NAMES.values()})


def _check_destination_is_a_bundle(root: Path) -> None:
    """Refuse to replace a directory this module did not write.

    ``os.replace`` cannot move a directory over a non-empty one, so re-saving
    over a bundle already worked; over somebody else's run directory would have
    taken emptying it first, and doing that quietly turns a save into an
    unrecoverable loss. So the destination must be absent, empty, or hold
    exactly what a bundle holds.
    """
    if not root.exists():
        return

    if not root.is_dir():
        raise ValueError(
            f"cannot write a bundle to {root}: a file of that name already "
            "exists. Move it aside, or save to a path that does not."
        )

    entries = list(root.iterdir())
    if not entries:
        return

    foreign = sorted(p.name for p in entries if p.name not in _BUNDLE_ENTRIES)
    if foreign:
        raise ValueError(
            f"refusing to write a bundle to {root}: it is not empty, and "
            f"{', '.join(repr(name) for name in foreign)} would be destroyed. "
            "A bundle holds only "
            f"{', '.join(sorted(_BUNDLE_ENTRIES))}, so that directory is not "
            "one. Move its contents aside, or save to a path that does not "
            "exist."
        )

    if MANIFEST_FILE not in {p.name for p in entries}:
        raise ValueError(
            f"refusing to write a bundle to {root}: it holds some of a "
            "bundle's files but no manifest.json, so nothing there could be "
            f"read back as one. Delete {root} yourself if that is fine."
        )


def _swap_into_place(staging: Path, root: Path, backup: Path) -> None:
    """Move a finished staging directory over the destination.

    ``os.replace`` cannot move a directory over a non-empty one, which is the
    re-save case, so the old bundle is renamed aside first. Same filesystem, so
    these are renames and not copies. :func:`_recover_interrupted_save` ran
    first and cleared any leftover, so ``backup`` is free here.
    """
    if root.exists():
        os.replace(root, backup)
    try:
        os.replace(staging, root)
    except BaseException:
        # Never leave the caller with nothing where a working bundle used to be.
        if backup.exists() and not root.exists():
            os.replace(backup, root)
        raise
    # Deliberately best-effort: the new bundle is in place, and raising here
    # would report a failure after a successful save. The next save to the same
    # path clears the leftover before it reuses the name.
    shutil.rmtree(backup, ignore_errors=True)


def _recover_interrupted_save(root: Path, backup: Path) -> None:
    """Clear a backup left by a save that was killed part-way through.

    A save that died after the first rename left no ``root`` and the old bundle
    as ``backup``; one that died after the second left a new bundle at ``root``
    and the old one as ``backup``. ``backup`` only ever holds what sat at
    ``root`` before this save started, so ``root`` is the newer of the two and
    the old one is the one to drop.

    Plain ``rmtree`` rather than ``ignore_errors``: a backup that survives is
    still non-empty, and the next ``os.replace`` onto it fails anyway, only with
    a message about a directory nobody chose.
    """
    if not backup.exists():
        return

    if root.exists():
        shutil.rmtree(backup)
    else:
        os.replace(backup, root)


def _clear_staging(staging: Path) -> None:
    """Remove a leftover staging directory from an interrupted save."""
    if staging.exists():
        shutil.rmtree(staging)


def _reference_logp(model: pm.Model) -> float | None:
    """Logp at the initial point, so :meth:`ModelBundle.validate` has something to compare.

    The point is computed with :data:`_REFERENCE_SEED`, because ``initval="prior"``
    draws and an unseeded reference would differ on every save. It returns None and
    says so when the model cannot produce one: a bundle whose reference is silently
    missing reads as "nothing to compare", which is easy to mistake for "nothing to
    worry about".
    """
    import warnings

    try:
        import numpy as np

        point = model.initial_point(random_seed=_REFERENCE_SEED)
        value = float(np.asarray(model.compile_logp()(point), dtype=float))
    except Exception as exc:
        warnings.warn(
            f"could not compute a reference logp for this model, so the bundle "
            f"will record reference_logp=null and validate() cannot check drift: "
            f"{type(exc).__name__}: {exc}",
            UserWarning,
            stacklevel=3,
        )
        return None
    # NaN and inf are not JSON, and a reference that cannot be compared is worse
    # than no reference at all.
    if not np.isfinite(value):
        warnings.warn(
            f"the reference logp for this model is {value}, which is not JSON "
            f"representable, so the bundle will record reference_logp=null and "
            f"validate() cannot check drift.",
            UserWarning,
            stacklevel=3,
        )
        return None
    return value


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
        if not (root / MODEL_FILE).exists():
            raise FileNotFoundError(
                f"{root} has a {MANIFEST_FILE} but no {MODEL_FILE}, so the model "
                f"is missing. The bundle was probably copied incompletely, or the "
                f"graph was written somewhere else on purpose; in the latter case "
                f"use ModelBundle.from_parts."
            )
        self._blob = (root / MODEL_FILE).read_bytes()
        self.path = root
        self.manifest = wrapped
        self._model: pm.Model | None = None

    @classmethod
    def from_parts(
        cls,
        manifest: dict | str | bytes,
        blob: bytes,
        *,
        path: str | Path = _IN_MEMORY,
    ) -> ModelBundle:
        """Build a bundle from a manifest and a graph you fetched yourself.

        The counterpart to :func:`build_manifest` and :func:`serialize_model`: use
        this when the bundle did not come from a local directory, such as an
        object store, a registry, or a database.

        Reading ``blob`` unpickles it, which runs code the same way
        ``pickle.load`` does, so treat anything you fetched from a remote store
        as untrusted until you know who wrote it.

        Parameters
        ----------
        manifest : dict or str or bytes
            The manifest, as a dict or as raw JSON.
        blob : bytes
            The serialized Model from :func:`serialize_model`.
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
                read_somewhere("model/model.cloudpickle"),
            )

            # embedded data needs the directory it was downloaded into
            bundle = ModelBundle.from_parts(
                manifest_bytes,
                blob_bytes,
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
        bundle._blob = blob
        bundle._model = None
        return bundle

    @property
    def data(self) -> DataRef | None:
        """The recorded data pointer, or None if the bundle makes no claim."""
        return self.manifest.data

    def resolve(self, ref: DataRef | None = None) -> str | Path:
        """Resolve a pointer to something openable.

        Data this bundle embedded is recorded relative to the bundle, so it moves
        with it and is resolved against ``self.path``. Anything else is the
        caller's own path or URL and is returned as written, with ``~`` expanded
        so it still means the same directory to the process that opens it.
        """
        ref = ref if ref is not None else self.data
        if ref is None:
            raise ValueError(f"{self.path} records no data pointer")
        if not ref.is_portable:
            return ref.location if ref.is_remote else ref.expanded()
        if str(self.path) == _IN_MEMORY:
            # Embedded data travels with the bundle, so a bundle assembled from
            # bytes in memory has nowhere to look. MLflow and friends hand the
            # bytes back alongside the directory they were fetched into; pass
            # that directory as ``path`` so relative pointers still resolve.
            raise ValueError(
                f"this bundle records data at {ref.location!r}, which is relative, "
                f"but it was built from memory rather than read from a directory. "
                f"Pass the directory the parts came from: "
                f"ModelBundle.from_parts(manifest, blob, path=that_directory)."
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
            _require_zarr("read")
            # consolidated=False because _write_embedded does not write the
            # .zarr_metadata sidecar. Asking for consolidated on a store that has
            # none buys no speed, only a failed attempt and a RuntimeWarning on
            # every read. For a store someone else did consolidate, this skips a
            # real speedup, which is the cost of not writing one ourselves.
            tree = xr.open_datatree(target, engine="zarr", consolidated=False)
        return tree if group is None else tree[group].to_dataset()

    def datatree(self):
        """Return the stored data as a whole DataTree, or None if there is none."""
        return None if self.data is None else self.open_data()

    @property
    def model(self) -> pm.Model:
        """The reconstructed Model.

        Dims round trip as-is, because the Model is unpickled rather than rebuilt
        from a graph. Rebuilding re-derives ``named_vars_to_dims`` and on a
        released pymc lands a list where the Model holds a tuple, so a restored
        model's declared dims would not compare equal to the original's.
        """
        if self._model is None:
            import cloudpickle

            _warn_untrusted_blob()
            self._model = cloudpickle.loads(self._blob)
        return self._model

    def validate(self, rtol: float = 1e-6, atol: float = 1e-8) -> ValidationResult:
        """Check this bundle really is the Model that was saved.

        Parameters
        ----------
        rtol : float
            Relative tolerance on the logp comparison. A logp on real data is
            often 1e4 to 1e6, where summation order alone across BLAS builds or
            thread counts moves it by more than any sensible absolute tolerance.
            Under ``floatX=float32`` the recorded value is itself noisy, so raise
            this if you validate such a bundle repeatedly.
        atol : float
            Absolute tolerance, for a logp near zero where a relative test would
            be meaningless.

        Notes
        -----
        The comparison is against the logp at the moment the bundle was written.
        Replacing shared variables afterwards, with ``pm.set_data``, changes the
        current logp without meaning the bundle is wrong, so a ``logp`` failure
        on a model you have re-pointed at new data is expected rather than a
        warning sign.

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
                np.asarray(
                    self.model.compile_logp()(
                        self.model.initial_point(random_seed=_REFERENCE_SEED)
                    )
                )
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
                result.checks["logp"] = _logp_close(actual, recorded, rtol, atol)
                if not result.checks["logp"]:
                    result.differences.append(
                        f"logp drifted: recorded {recorded!r}, now {actual!r}"
                    )
                    result.notes.append(
                        "the reference logp is the value at the time this bundle was "
                        "written. Changing shared variables since then, with "
                        "pm.set_data, moves the current value without meaning the "
                        "bundle is damaged, so check whether you did that before "
                        "treating this as corruption."
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

    Reading the manifest and the data is safe. The Model is only unpickled when
    you ask for :attr:`ModelBundle.model`, and that step runs code the same way
    ``pickle.load`` does, so :attr:`ModelBundle.model` warns.

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

    Unpickling the Model runs code, so only call this on a bundle you
    produced yourself or otherwise trust; loading runs code the same way
    ``pickle.load`` does.

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
    """Node count and graph depth.

    Walked in topological order rather than recursed, because depth is exactly the
    quantity that scales with graph size: a model with a few thousand chained
    additions recursed until the stack ran out and took save_model with it, while
    ``compile_logp`` handled the same graph without complaint.
    """
    depths: dict = {}
    for node in fgraph.toposort():
        # Inputs without an entry are leaves or graph inputs, which the recursive
        # definition counted as 0.
        node_depth = 1 + max((depths.get(inp, 0) for inp in node.inputs), default=0)
        for output in node.outputs:
            depths[output] = node_depth

    return (
        len(fgraph.apply_nodes),
        max((depths.get(output, 0) for output in fgraph.outputs), default=0),
    )


def _names(sequence) -> list[str]:
    """Model collections mix treelists of Variables with treedicts keyed by name."""
    return [v if isinstance(v, str) else v.name for v in sequence]


def _python_version() -> str:
    """Return the running Python version, which cloudpickled bytecode depends on."""
    import platform

    return platform.python_version()


def _floatX() -> str:
    """Return pytensor's floatX, which bounds how precise a logp can be."""
    import pytensor

    return str(pytensor.config.floatX)


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
