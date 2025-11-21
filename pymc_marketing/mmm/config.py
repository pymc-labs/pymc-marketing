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
    """Configuration dictionary for MMM plotting settings.

    Global configuration object that controls MMM plotting behavior including
    backend selection and version control. Modeled after ArviZ's rcParams pattern.

    Available Configuration Keys
    ----------------------------

    **plot.backend** : str, default="matplotlib"
        Plotting backend to use for all plots in MMMPlotSuite. Options:

        * ``"matplotlib"`` - Static plots, publication-quality, widest compatibility
        * ``"plotly"`` - Interactive plots with hover tooltips and zoom
        * ``"bokeh"`` - Interactive plots with rich interactions

        Can be overridden per-method using the ``backend`` parameter.

        .. versionadded:: 0.18.0

    **plot.show_warnings** : bool, default=True
        Whether to show deprecation and other warnings from the plotting suite.

        .. versionadded:: 0.18.0

    **plot.use_v2** : bool, default=False
        Whether to use new arviz_plots-based plotting suite vs legacy suite.

        * ``False`` (default in v0.18.0): Use legacy matplotlib-only suite
        * ``True``: Use new multi-backend arviz_plots-based suite

        This flag controls which suite is returned by ``MMM.plot`` property.

        .. versionadded:: 0.18.0

        .. versionchanged:: 0.19.0
           Default will change to True (new suite becomes default).

        .. deprecated:: 0.20.0
           This flag will be removed as legacy suite is removed.

    Examples
    --------
    Set plotting backend globally:

    >>> from pymc_marketing.mmm import mmm_config
    >>> mmm_config["plot.backend"] = "plotly"
    >>> # All plots now use plotly by default
    >>> mmm = MMM(...)
    >>> mmm.fit(X, y)
    >>> pc = mmm.plot.posterior_predictive()  # Uses plotly
    >>> pc.show()

    Enable new plotting suite (v2):

    >>> mmm_config["plot.use_v2"] = True
    >>> # Now using arviz_plots-based multi-backend suite
    >>> mmm = MMM(...)
    >>> mmm.fit(X, y)
    >>> pc = mmm.plot.contributions_over_time(var=["intercept"])
    >>> pc.show()

    Suppress warnings:

    >>> mmm_config["plot.show_warnings"] = False

    Reset to defaults:

    >>> mmm_config.reset()
    >>> mmm_config["plot.backend"]
    'matplotlib'

    Context manager pattern for temporary config changes:

    >>> original = mmm_config["plot.backend"]
    >>> try:
    ...     mmm_config["plot.backend"] = "plotly"
    ...     # Use plotly for this section
    ...     pc = mmm.plot.posterior_predictive()
    ...     pc.show()
    ... finally:
    ...     mmm_config["plot.backend"] = original

    See Also
    --------
    MMM.plot : Property that returns appropriate plot suite based on config
    MMMPlotSuite : New multi-backend plotting suite
    LegacyMMMPlotSuite : Legacy matplotlib-only suite

    Notes
    -----
    Configuration changes affect all subsequent plot calls globally unless
    overridden at the method level using the ``backend`` parameter.

    The configuration is a singleton - changes affect all MMM instances in
    the current Python session.
    """

    _defaults = {
        "plot.backend": "matplotlib",
        "plot.show_warnings": True,
        "plot.use_v2": False,  # Use new arviz_plots-based suite (False = legacy suite for backward compatibility)
    }

    VALID_KEYS = set(_defaults.keys())

    def __init__(self):
        super().__init__(self._defaults)

    def __setitem__(self, key, value):
        """Set config value with validation for key and backend."""
        if key not in self.VALID_KEYS:
            warnings.warn(
                f"Invalid config key '{key}'. Valid keys are: {sorted(self.VALID_KEYS)}. "
                f"Setting anyway, but this key may not be recognized.",
                UserWarning,
                stacklevel=2,
            )
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
