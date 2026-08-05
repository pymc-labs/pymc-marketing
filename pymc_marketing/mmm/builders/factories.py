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
"""Generic recursive factory for the MMM YAML schema."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from typing import Any

from pymc_extras.deserialize import DeserializableError, deserialize

from pymc_marketing.model_config import ModelConfigError

# Importing ``special_priors`` registers the deserializer for the flat YAML
# ``{"distribution": ...}`` prior format, which pymc-extras' built-in
# deserializer does not handle.
import pymc_marketing.special_priors  # noqa: F401  # isort: skip

# Optional short-name registry -------------------------------------------------
REGISTRY: dict[str, Any] = {
    # "Prior": pymc_extras.prior.Prior,   # <— example of a whitelisted alias
}

KNOWN_SPEC_KEYS = frozenset({"class", "kwargs", "args"})

# -----------------------------------------------------------------------------


def locate(qualname: str) -> Any:
    """
    Resolve *qualname* to a Python callable.

    Parameters
    ----------
    qualname : str
        Either a dotted import path ('pkg.mod.Class') or a key in REGISTRY.
    """
    # Check if qualname is a dictionary (which would cause the error)
    if not isinstance(qualname, str):
        raise TypeError(
            f"Expected string for qualname but got {type(qualname).__name__}: {qualname}"
        )

    if qualname in REGISTRY:
        return REGISTRY[qualname]

    module, _, obj_name = qualname.rpartition(".")
    if not module:
        raise ValueError(
            f"Cannot locate '{qualname}'. "
            "Provide a fully-qualified name or add it to REGISTRY."
        )
    module_obj = importlib.import_module(module)
    return getattr(module_obj, obj_name)


@contextmanager
def naming(key: str) -> Iterator[None]:
    """Name *key* on any deserialization failure raised inside the block.

    YAML is the entry point most likely to be hand-edited, and a bare
    ``DeserializableError`` only dumps the offending sub-dict, which is hard to
    trace back to a config entry. ``ModelConfigError`` used to name every bad
    key before dict-format priors were removed; this preserves that.

    ``DeserializableError`` reports "use register_deserialization to add a
    mapping", which is the wrong advice when a deserializer did match and
    failed for some other reason. In that case ``__cause__`` holds the actual
    reason (e.g. ``PyMC doesn't have a distribution of name 'Nrmal'``), so
    surface that instead.
    """
    try:
        yield
    except DeserializableError as err:
        raise ModelConfigError(f"Parameter {key}: {err.__cause__ or err}") from err


def build(spec: Mapping[str, Any]) -> Any:
    """
    Instantiate the object described by *spec*.

    Notes
    -----
    Recognised keys
    * class : str   (mandatory)
    * kwargs : dict  (optional)
    * args   : list  (optional positional arguments)
    """
    if not isinstance(spec["class"], str):
        raise TypeError(
            f"Expected string for 'class' but got {type(spec['class']).__name__}: {spec['class']}"
        )

    unknown_keys = set(spec.keys()) - KNOWN_SPEC_KEYS
    if unknown_keys:
        warnings.warn(
            f"Unknown keys {unknown_keys} in build spec for "
            f"'{spec['class']}'. Only {sorted(KNOWN_SPEC_KEYS)} are "
            f"recognised; other keys are ignored.",
            UserWarning,
            stacklevel=2,
        )

    cls = locate(spec["class"])

    raw_kwargs: MutableMapping[str, Any] = dict(spec.get("kwargs", {}))
    raw_args: Sequence[Any] = raw_kwargs.pop("args", spec.get("args", ()))

    # Handle specific kwargs that should be processed differently
    special_processing_keys = ["priors", "prior", "model_config"]

    # Convert list dimensions to tuples for model or effect classes
    if "dims" in raw_kwargs and isinstance(raw_kwargs["dims"], list):
        try:
            raw_kwargs["dims"] = tuple(raw_kwargs["dims"])
        except Exception as e:
            print(f"Warning: Could not convert dims to tuple: {e}")

    kwargs = {}
    for k, v in raw_kwargs.items():
        if k in special_processing_keys:
            # Handle priors and prior differently
            if isinstance(v, dict):
                if k == "priors":
                    priors_dict = {}
                    for prior_key, prior_value in v.items():
                        if isinstance(prior_value, dict):
                            if "class" in prior_value:
                                priors_dict[prior_key] = build(prior_value)
                            else:
                                with naming(prior_key):
                                    priors_dict[prior_key] = deserialize(prior_value)
                        else:
                            priors_dict[prior_key] = prior_value
                    kwargs[k] = priors_dict
                elif k == "prior" and "distribution" in v:
                    with naming(k):
                        kwargs[k] = deserialize(v)
                elif k == "model_config":
                    # Each entry may be a prior spec ({"distribution": ...}),
                    # an HSGP/other config dict, or a plain value. ``resolve``
                    # deserializes prior specs and leaves everything else as-is,
                    # since ``parse_model_config`` no longer converts dicts.
                    model_config = {}
                    for mk, mv in v.items():
                        with naming(mk):
                            model_config[mk] = resolve(mv)
                    kwargs[k] = model_config
                else:
                    with naming(k):
                        kwargs[k] = resolve(v)
            else:
                with naming(k):
                    kwargs[k] = resolve(v)
        else:
            # --- recurse into nested objects for other items -----------------------------------------
            with naming(k):
                kwargs[k] = resolve(v)

    args = []
    for i, v in enumerate(raw_args):
        with naming(f"args[{i}]"):
            args.append(resolve(v))

    return cls(*args, **kwargs)


def resolve(value):
    """
    Resolve a value by recursively building nested objects.

    This is a helper function for build.
    """
    if isinstance(value, Mapping):
        if "class" in value:
            return build(value)
        # "dist" is the pymc-extras serialization key; "distribution" is the
        # flat YAML form handled by ``special_priors.deserialize_alternative_prior``.
        # Require a string value so non-prior mappings that merely contain such
        # a key are not misrouted to the prior deserializer.
        if (
            isinstance(value.get("distribution"), str)
            or isinstance(value.get("dist"), str)
            or "special_prior" in value
        ):
            return deserialize(value)

    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], Mapping)
        and "class" in value[0]
    ):
        return [build(v) for v in value]

    return value
