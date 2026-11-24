#   Copyright 2022 - 2026 The PyMC Labs Developers
#   Licensed under the Apache License, Version 2.0

"""Write the PyTensor compile-cache version key for GitHub Actions caching.

The key records the installed versions of the packages that determine
compiled artifacts, plus the portable-target pins. It is intentionally
language-agnostic about the interpreter version: the numba cache suffixes
cache files with the interpreter version and the C compiledir embeds it in
its directory name, so entries self-select at load time.

Requires PYTENSOR_FLAGS and NUMBA_CPU_NAME in the environment (set at the
job level) so that the portable-target pins are part of the version hash:
they change compile flags and cache keys, and omitting them would make
new-style compilations unpublishable over old-style entries.
"""

import importlib.metadata as md
import os

pkgs = ["pytensor", "numba", "llvmlite", "numpy", "scipy"]


def ver(pkg: str) -> str:
    """Return the installed version of ``pkg``, or ``missing``."""
    try:
        return md.version(pkg)
    except Exception:
        return "missing"


pflags = os.environ["PYTENSOR_FLAGS"]
ncpu = os.environ["NUMBA_CPU_NAME"]

with open(".pytensor-cache-key", "w") as f:
    for pkg in pkgs:
        f.write(f"{pkg}={ver(pkg)}\n")
    f.write(f"pytensor_flags={pflags}\n")
    f.write(f"numba_cpu={ncpu}\n")
