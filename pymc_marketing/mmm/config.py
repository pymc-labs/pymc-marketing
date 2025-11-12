#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Configuration management for MMM plotting."""

import warnings

VALID_BACKENDS = {"matplotlib", "plotly", "bokeh"}


class MMMConfig(dict):
    """
    Configuration dictionary for MMM plotting settings.

    Provides backend configuration with validation and reset functionality.
    Modeled after ArviZ's rcParams pattern.

    Examples
    --------
    >>> from pymc_marketing.mmm import mmm_config
    >>> mmm_config["plot.backend"] = "plotly"
    >>> mmm_config["plot.backend"]
    'plotly'
    >>> mmm_config.reset()
    >>> mmm_config["plot.backend"]
    'matplotlib'
    """

    _defaults = {
        "plot.backend": "matplotlib",
        "plot.show_warnings": True,
    }

    def __init__(self):
        super().__init__(self._defaults)

    def __setitem__(self, key, value):
        """Set config value with validation for backend."""
        if key == "plot.backend":
            if value not in VALID_BACKENDS:
                warnings.warn(
                    f"Invalid backend '{value}'. Valid backends are: {VALID_BACKENDS}. "
                    f"Setting anyway, but plotting may fail.",
                    UserWarning,
                    stacklevel=2,
                )
        super().__setitem__(key, value)

    def reset(self):
        """Reset all configuration to default values."""
        self.clear()
        self.update(self._defaults)


# Global config instance
mmm_config = MMMConfig()
