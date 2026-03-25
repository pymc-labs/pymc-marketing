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
"""PyMC-Marketing."""

# Suppress pymc.dims experimental warning before any module imports it
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore", message="The `pymc.dims` module is experimental")
    import pymc.dims  # noqa: F401

del _warnings

# Load the data accessor
import pymc_marketing.data.fivetran  # noqa: E402

# Register SpecialPrior deserializers (LogNormalPrior, LaplacePrior, etc.)
import pymc_marketing.special_priors  # noqa: E402, F401
from pymc_marketing import bass, clv, customer_choice, mmm  # noqa: E402
from pymc_marketing.version import __version__  # noqa: E402

__all__ = ["__version__", "bass", "clv", "customer_choice", "mmm"]
