#   Copyright 2022 - 2026 The PyMC Labs Developers
#   Licensed under the Apache License, Version 2.0

"""Write the PyTensor compile-cache version key for GitHub Actions caching.

The key records the installed versions of the packages that determine
compiled artifacts, plus the portable-target pins. It records the interpreter
version: the numba cache suffixes cache files with the interpreter version
and the C compiledir embeds it, so artifacts from different interpreter
versions can never be reused, and mixing them in one pool only wastes
restore bandwidth.

Requires PYTENSOR_FLAGS and NUMBA_CPU_NAME in the environment (set at the
job level) so that the portable-target pins are part of the version hash:
the pins change compile flags and cache keys, so leaving them out would
let a cache publish entries that later runs cannot use.
"""

import importlib.metadata as md
import os
import sys

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
    f.write(f"python={sys.version_info[0]}.{sys.version_info[1]}\n")
    for pkg in pkgs:
        f.write(f"{pkg}={ver(pkg)}\n")
    f.write(f"pytensor_flags={pflags}\n")
    f.write(f"numba_cpu={ncpu}\n")
