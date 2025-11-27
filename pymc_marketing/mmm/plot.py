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
"""MMM related plotting class.

Examples
--------
Quickstart with MMM:

.. code-block:: python

    from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
    from pymc_marketing.mmm.multidimensional import MMM
    import pandas as pd

    # Minimal dataset
    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=12, freq="W-MON"),
            "C1": [100, 120, 90, 110, 105, 115, 98, 102, 108, 111, 97, 109],
            "C2": [80, 70, 95, 85, 90, 88, 92, 94, 91, 89, 93, 87],
        }
    )
    y = pd.Series(
        [230, 260, 220, 240, 245, 255, 235, 238, 242, 246, 233, 249], name="y"
    )

    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        target_column="y",
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )
    mmm.fit(X, y)
    mmm.sample_posterior_predictive(X)

    # Posterior predictive time series
    _ = mmm.plot.posterior_predictive(var="y", hdi_prob=0.9)

    # Posterior contributions over time (e.g., channel_contribution)
    _ = mmm.plot.contributions_over_time(var=["channel_contribution"], hdi_prob=0.9)

    # Channel saturation scatter plot (scaled space by default)
    _ = mmm.plot.saturation_scatterplot(original_scale=False)

Wrap a custom PyMC model
--------

Requirements

- posterior_predictive plots: an `az.InferenceData` with a `posterior_predictive` group
  containing the variable(s) you want to plot with a `date` coordinate.
- contributions_over_time plots: a `posterior` group with time‑series variables (with `date`).
- saturation plots: a `constant_data` dataset with variables:
  - `channel_data`: dims include `("date", "channel", ...)`
  - `channel_scale`: dims include `("channel", ...)`
  - `target_scale`: scalar or broadcastable to the curve dims
  and a `posterior` variable named `channel_contribution` (or
  `channel_contribution_original_scale` if plotting `original_scale=True`).

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pymc as pm
    from pymc_marketing.mmm.plot import MMMPlotSuite

    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    y_obs = np.random.normal(size=len(dates))

    with pm.Model(coords={"date": dates}):
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", 0.0, sigma, observed=y_obs, dims="date")

        idata = pm.sample_prior_predictive(random_seed=1)
        idata.extend(pm.sample(draws=200, chains=2, tune=200, random_seed=1))
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=1))

    plot = MMMPlotSuite(idata)
    _ = plot.posterior_predictive(var="y", hdi_prob=0.9)

Custom contributions_over_time
--------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import pymc as pm
    from pymc_marketing.mmm.plot import MMMPlotSuite

    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    x = np.linspace(0, 2 * np.pi, len(dates))
    series = np.sin(x)

    with pm.Model(coords={"date": dates}):
        pm.Deterministic("component", series, dims="date")
        idata = pm.sample_prior_predictive(random_seed=2)
        idata.extend(pm.sample(draws=50, chains=1, tune=0, random_seed=2))

    plot = MMMPlotSuite(idata)
    _ = plot.contributions_over_time(var=["component"], hdi_prob=0.9)

Saturation plots with a custom model
--------

.. code-block:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    import pymc as pm
    from pymc_marketing.mmm.plot import MMMPlotSuite

    dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
    channels = ["C1", "C2"]

    # Create constant_data required for saturation plots
    channel_data = xr.DataArray(
        np.random.rand(len(dates), len(channels)),
        dims=("date", "channel"),
        coords={"date": dates, "channel": channels},
        name="channel_data",
    )
    channel_scale = xr.DataArray(
        np.ones(len(channels)),
        dims=("channel",),
        coords={"channel": channels},
        name="channel_scale",
    )
    target_scale = xr.DataArray(1.0, name="target_scale")

    # Build a toy model that yields a matching posterior var
    with pm.Model(coords={"date": dates, "channel": channels}):
        # A fake contribution over time per channel (dims must include date & channel)
        contrib = pm.Normal("channel_contribution", 0.0, 1.0, dims=("date", "channel"))

        idata = pm.sample_prior_predictive(random_seed=3)
        idata.extend(pm.sample(draws=50, chains=1, tune=0, random_seed=3))

    # Attach constant_data to idata
    idata.constant_data = xr.Dataset(
        {
            "channel_data": channel_data,
            "channel_scale": channel_scale,
            "target_scale": target_scale,
        }
    )

    plot = MMMPlotSuite(idata)
    _ = plot.saturation_scatterplot(original_scale=False)

Notes
-----
- `MMM` exposes this suite via the `mmm.plot` property, which internally passes the model's
  `idata` into `MMMPlotSuite`.
- Any PyMC model can use `MMMPlotSuite` directly if its `InferenceData` contains the needed
  groups/variables described above.
"""

import itertools

import arviz as az
import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection

from pymc_marketing.mmm.config import mmm_config

__all__ = ["MMMPlotSuite"]

WIDTH_PER_COL: float = 10.0
HEIGHT_PER_ROW: float = 4.0


class MMMPlotSuite:
    """Media Mix Model Plot Suite.

    Provides methods for visualizing the posterior predictive distribution,
    contributions over time, and saturation curves for a Media Mix Model.
    """

    def __init__(
        self,
        idata: xr.Dataset | az.InferenceData,
    ):
        self.idata = idata

    def _get_additional_dim_combinations(
        self,
        data: xr.Dataset,
        variable: str,
        ignored_dims: set[str],
    ) -> tuple[list[str], list[tuple]]:
        """Identify dimensions to plot over and get their coordinate combinations."""
        if variable not in data:
            raise ValueError(f"Variable '{variable}' not found in the dataset.")

        all_dims = list(data[variable].dims)
        additional_dims = [d for d in all_dims if d not in ignored_dims]

        if additional_dims:
            additional_coords = [data.coords[d].values for d in additional_dims]
            dim_combinations = list(itertools.product(*additional_coords))
        else:
            # If no extra dims, just treat as a single combination
            dim_combinations = [()]

        return additional_dims, dim_combinations

    def _get_posterior_predictive_data(
        self,
        idata: xr.Dataset | None,
    ) -> xr.Dataset:
        """Retrieve the posterior_predictive group from either provided or self.idata."""
        if idata is not None:
            return idata

        # Otherwise, check if self.idata has posterior_predictive
        if (
            not hasattr(self.idata, "posterior_predictive")  # type: ignore
            or self.idata.posterior_predictive is None  # type: ignore
        ):
            raise ValueError(
                "No posterior_predictive data found in 'self.idata'. "
                "Please run 'MMM.sample_posterior_predictive()' or provide "
                "an external 'idata' argument."
            )
        return self.idata.posterior_predictive  # type: ignore

    def _validate_dims(
        self,
        dims: dict[str, str | int | list],
        all_dims: list[str],
    ) -> None:
        """Validate that provided dims exist in the model's dimensions and values."""
        if dims:
            for key, val in dims.items():
                if key not in all_dims:
                    raise ValueError(
                        f"Dimension '{key}' not found in idata dimensions."
                    )
                valid_values = self.idata.posterior.coords[key].values
                if isinstance(val, (list, tuple, np.ndarray)):
                    for v in val:
                        if v not in valid_values:
                            raise ValueError(
                                f"Value '{v}' not found in dimension '{key}'."
                            )
                else:
                    if val not in valid_values:
                        raise ValueError(
                            f"Value '{val}' not found in dimension '{key}'."
                        )

    def _dim_list_handler(
        self, dims: dict[str, str | int | list] | None
    ) -> tuple[list[str], list[tuple]]:
        """Extract keys, values, and all combinations for list-valued dims."""
        dims_lists = {
            k: v
            for k, v in (dims or {}).items()
            if isinstance(v, (list, tuple, np.ndarray))
        }
        if dims_lists:
            dims_keys = list(dims_lists.keys())
            dims_values = [
                v if isinstance(v, (list, tuple, np.ndarray)) else [v]
                for v in dims_lists.values()
            ]
            dims_combos = list(itertools.product(*dims_values))
        else:
            dims_keys = []
            dims_combos = [()]
        return dims_keys, dims_combos

    def _resolve_backend(self, backend: str | None) -> str:
        """Resolve backend parameter to actual backend string."""
        return backend or mmm_config["plot.backend"]

    def _get_data_or_fallback(
        self,
        data: xr.Dataset | None,
        idata_attr: str,
        data_name: str,
    ) -> xr.Dataset:
        """Get data from parameter or fall back to self.idata attribute.

        Parameters
        ----------
        data : xr.Dataset or None
            Data provided by user.
        idata_attr : str
            Attribute name on self.idata to use as fallback (e.g., "posterior").
        data_name : str
            Human-readable name for error messages (e.g., "posterior data").

        Returns
        -------
        xr.Dataset
            The data to use.

        Raises
        ------
        ValueError
            If data is None and self.idata doesn't have the required attribute.
        """
        if data is None:
            if not hasattr(self.idata, idata_attr):
                raise ValueError(
                    f"No {data_name} found in 'self.idata' and no 'data' argument provided. "
                    f"Please ensure 'self.idata' contains a '{idata_attr}' group or provide 'data' explicitly."
                )
            data = getattr(self.idata, idata_attr)
        return data

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self,
        var: str | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot posterior predictive distributions over time.

        Visualizes posterior predictive samples as time series, showing the median
        line and highest density interval (HDI) bands. Useful for checking model fit
        and understanding prediction uncertainty.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        var : str, optional
            Variable name to plot from posterior_predictive group. If None, uses "y".
        idata : xr.Dataset, optional
            Dataset containing posterior predictive samples with a "date" coordinate.
            If None, uses self.idata.posterior_predictive.

            This parameter allows:
            - Testing with mock data without modifying self.idata
            - Plotting external posterior predictive samples
            - Comparing different model fits side-by-side
        hdi_prob : float, default 0.85
            Probability mass for HDI interval (between 0 and 1).
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default is "matplotlib".

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If no posterior_predictive data found in self.idata and no idata provided.
        ValueError
            If hdi_prob is not between 0 and 1.

        See Also
        --------
        LegacyMMMPlotSuite.posterior_predictive : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes)
        - Different interface for saving and displaying plots

        Examples
        --------
        Basic usage:

        .. code-block:: python

            mmm.sample_posterior_predictive(X)
            pc = mmm.plot.posterior_predictive()
            pc.show()

        Plot with different HDI probability:

        .. code-block:: python

            pc = mmm.plot.posterior_predictive(hdi_prob=0.94)
            pc.show()

        Save to file:

        .. code-block:: python

            pc = mmm.plot.posterior_predictive()
            pc.save("posterior_predictive.png")

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.posterior_predictive(backend="plotly")
            pc.show()

        Provide explicit data:

        .. code-block:: python

            external_pp = xr.Dataset(...)  # Custom posterior predictive
            pc = mmm.plot.posterior_predictive(idata=external_pp)
            pc.show()

        Direct instantiation pattern:

        .. code-block:: python

            from pymc_marketing.mmm.plot import MMMPlotSuite

            mps = MMMPlotSuite(custom_idata)
            pc = mps.posterior_predictive()
            pc.show()
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")

        # Resolve backend
        backend = self._resolve_backend(backend)

        # 1. Retrieve or validate posterior_predictive data
        pp_data = self._get_posterior_predictive_data(idata)

        # 2. Determine variable to plot
        if var is None:
            var = "y"
        main_var = var

        # 3. Identify additional dims & get all combos
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, _ = self._get_additional_dim_combinations(
            data=pp_data, variable=main_var, ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        pc = azp.PlotCollection.wrap(
            pp_data[main_var].to_dataset(),
            cols=additional_dims,
            col_wrap=1,
            figure_kwargs={
                "sharex": True,
            },
            backend=backend,
        )

        # plot hdi
        hdi = pp_data.azstats.hdi(hdi_prob)
        pc.map(
            azp.visuals.fill_between_y,
            x=pp_data["date"],
            y_bottom=hdi.sel(ci_bound="lower"),
            y_top=hdi.sel(ci_bound="upper"),
            alpha=0.2,
            color="C0",
        )

        # plot median line
        pc.map(
            azp.visuals.line_xy,
            x=pp_data["date"],
            y=pp_data.median(dim=["chain", "draw"]),
            color="C0",
        )

        # add labels
        pc.map(azp.visuals.labelled_x, text="Date")
        pc.map(azp.visuals.labelled_y, text="Posterior Predictive")
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        )
        return pc

    def contributions_over_time(
        self,
        var: list[str],
        data: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot time-series contributions for specified variables.

        Visualizes how variables contribute over time, showing the median line and
        HDI bands. Useful for understanding channel contributions, intercepts, or
        other time-varying effects in your model.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        var : list of str
            Variable names to plot from the posterior group. Must have a "date" dimension.
            Examples: ["channel_contribution"], ["intercept"], ["channel_contribution", "intercept"].
        data : xr.Dataset, optional
            Dataset containing posterior data with variables in `var`.
            If None, uses self.idata.posterior.

            .. versionadded:: 0.18.0
               Added data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock data without modifying self.idata
            - Plotting external results not stored in self.idata
            - Comparing different posterior samples side-by-side
            - Avoiding unintended side effects on self.idata
        hdi_prob : float, default 0.85
            Probability mass for HDI interval (between 0 and 1).
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Keys are dimension names, values are either:
            - Single value: {"country": "US", "user_type": "new"}
            - List of values: {"country": ["US", "UK"]}

            If provided, only the selected slice(s) will be plotted.
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default is "matplotlib".

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If hdi_prob is not between 0 and 1.
        ValueError
            If no posterior data found in self.idata and no data argument provided.
        ValueError
            If any variable in `var` not found in data.

        See Also
        --------
        LegacyMMMPlotSuite.contributions_over_time : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes)
        - Variable names must be passed in a list (was already list in legacy)

        Examples
        --------
        Basic usage - plot channel contributions:

        .. code-block:: python

            mmm.fit(X, y)
            pc = mmm.plot.contributions_over_time(var=["channel_contribution"])
            pc.show()

        Plot multiple variables together:

        .. code-block:: python

            pc = mmm.plot.contributions_over_time(
                var=["channel_contribution", "intercept"]
            )
            pc.show()

        Filter by dimension:

        .. code-block:: python

            pc = mmm.plot.contributions_over_time(
                var=["channel_contribution"], dims={"geo": "US"}
            )
            pc.show()

        Filter with multiple dimension values:

        .. code-block:: python

            pc = mmm.plot.contributions_over_time(
                var=["channel_contribution"], dims={"geo": ["US", "UK"]}
            )
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.contributions_over_time(
                var=["channel_contribution"], backend="plotly"
            )
            pc.show()

        Provide explicit data (option 1 - via data parameter):

        .. code-block:: python

            custom_posterior = xr.Dataset(...)
            pc = mmm.plot.contributions_over_time(
                var=["my_contribution"], data=custom_posterior
            )
            pc.show()

        Provide explicit data (option 2 - direct instantiation):

        .. code-block:: python

            from pymc_marketing.mmm.plot import MMMPlotSuite

            mps = MMMPlotSuite(custom_idata)
            pc = mps.contributions_over_time(var=["my_contribution"])
            pc.show()
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")

        # Get data with fallback to self.idata.posterior
        data = self._get_data_or_fallback(data, "posterior", "posterior data")

        # Validate data has the required variables
        missing_vars = [v for v in var if v not in data]
        if missing_vars:
            raise ValueError(
                f"Variables {missing_vars} not found in data. "
                f"Available variables: {list(data.data_vars)}"
            )

        # Resolve backend
        backend = self._resolve_backend(backend)

        main_var = var[0]
        ignored_dims = {"chain", "draw", "date"}
        da = data[var]

        # Apply dims filtering if provided
        if dims:
            self._validate_dims(dims, list(da[main_var].dims))
            for dim_name, dim_value in dims.items():
                if isinstance(dim_value, (list, tuple, np.ndarray)):
                    da = da.sel({dim_name: dim_value})
                else:
                    da = da.sel({dim_name: dim_value})

        additional_dims, _ = self._get_additional_dim_combinations(
            data=da, variable=main_var, ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        pc = azp.PlotCollection.wrap(
            da,
            cols=additional_dims,
            col_wrap=1,
            figure_kwargs={
                "sharex": True,
            },
            backend=backend,
        )

        # plot hdi
        hdi = da.azstats.hdi(hdi_prob)
        pc.map(
            azp.visuals.fill_between_y,
            x=da["date"],
            y_bottom=hdi.sel(ci_bound="lower"),
            y_top=hdi.sel(ci_bound="upper"),
            alpha=0.2,
            color="C0",
        )

        # plot median line
        pc.map(
            azp.visuals.line_xy,
            x=da["date"],
            y=da.median(dim=["chain", "draw"]),
            color="C0",
        )

        # add labels
        pc.map(azp.visuals.labelled_x, text="Date")
        pc.map(azp.visuals.labelled_y, text="Posterior Value")
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        )

        return pc

    def saturation_scatterplot(
        self,
        original_scale: bool = False,
        constant_data: xr.Dataset | None = None,
        posterior_data: xr.Dataset | None = None,
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot saturation scatter plot showing channel spend vs contributions.

        Creates scatter plots of actual channel spend (X-axis) against channel
        contributions (Y-axis), one subplot per channel. Useful for understanding
        the saturation behavior and diminishing returns of each marketing channel.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        original_scale : bool, default False
            Whether to plot in original scale (True) or scaled space (False).
            If True, requires channel_contribution_original_scale in posterior.
        constant_data : xr.Dataset, optional
            Dataset containing constant_data group with required variables:
            - 'channel_data': Channel spend data (dims include "date", "channel")
            - 'channel_scale': Scaling factor per channel (if original_scale=True)
            - 'target_scale': Target scaling factor (if original_scale=True)

            If None, uses self.idata.constant_data.

            .. versionadded:: 0.18.0
               Added constant_data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock constant data
            - Plotting with alternative scaling factors
            - Comparing different data scenarios
        posterior_data : xr.Dataset, optional
            Dataset containing posterior group with channel contribution variables.
            Must contain 'channel_contribution' or 'channel_contribution_original_scale'.
            If None, uses self.idata.posterior.

            .. versionadded:: 0.18.0
               Added posterior_data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock posterior samples
            - Plotting external inference results
            - Comparing different model fits
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Examples:
            - {"geo": "US"} - Single value
            - {"geo": ["US", "UK"]} - Multiple values

            If provided, only the selected slice(s) will be plotted.
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default is "matplotlib".

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If required data not found in self.idata and not provided explicitly.
        ValueError
            If 'channel_data' not found in constant_data.
        ValueError
            If original_scale=True but channel_contribution_original_scale not in posterior.

        See Also
        --------
        saturation_curves : Add posterior predictive curves to this scatter plot
        LegacyMMMPlotSuite.saturation_scatterplot : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes)
        - Lost **kwargs for matplotlib customization (use backend-specific methods)
        - Different grid layout algorithm

        Examples
        --------
        Basic usage (scaled space):

        .. code-block:: python

            mmm.fit(X, y)
            pc = mmm.plot.saturation_scatterplot()
            pc.show()

        Plot in original scale:

        .. code-block:: python

            mmm.add_original_scale_contribution_variable(var=["channel_contribution"])
            pc = mmm.plot.saturation_scatterplot(original_scale=True)
            pc.show()

        Filter by dimension:

        .. code-block:: python

            pc = mmm.plot.saturation_scatterplot(dims={"geo": "US"})
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.saturation_scatterplot(backend="plotly")
            pc.show()

        Provide explicit data:

        .. code-block:: python

            custom_constant = xr.Dataset(...)
            custom_posterior = xr.Dataset(...)
            pc = mmm.plot.saturation_scatterplot(
                constant_data=custom_constant, posterior_data=custom_posterior
            )
            pc.show()
        """
        # Resolve backend
        backend = self._resolve_backend(backend)

        # Get constant_data and posterior_data with fallback
        constant_data = self._get_data_or_fallback(
            constant_data, "constant_data", "constant data"
        )
        posterior_data = self._get_data_or_fallback(
            posterior_data, "posterior", "posterior data"
        )

        # Validate required variables exist
        if "channel_data" not in constant_data:
            raise ValueError(
                "'channel_data' variable not found in constant_data. "
                f"Available variables: {list(constant_data.data_vars)}"
            )

        # Identify additional dimensions beyond 'date' and 'channel'
        cdims = constant_data.channel_data.dims
        additional_dims = [dim for dim in cdims if dim not in ("date", "channel")]

        # Validate dims and remove filtered dims from additional_dims
        if dims:
            self._validate_dims(dims, list(constant_data.channel_data.dims))
            additional_dims = [d for d in additional_dims if d not in dims]
        else:
            self._validate_dims({}, list(constant_data.channel_data.dims))

        channel_contribution = (
            "channel_contribution_original_scale"
            if original_scale
            else "channel_contribution"
        )

        if channel_contribution not in posterior_data:
            raise ValueError(
                f"""No posterior.{channel_contribution} data found in posterior_data. \n
                Add a original scale deterministic:\n
                mmm.add_original_scale_contribution_variable(\n
                var=[\n
                \"channel_contribution\",\n
                ...\n
                ]\n
                )\n
                """
            )

        # Apply dims filtering to channel_data and channel_contribution
        channel_data = constant_data.channel_data
        channel_contrib = posterior_data[channel_contribution]

        if dims:
            for dim_name, dim_value in dims.items():
                if isinstance(dim_value, (list, tuple, np.ndarray)):
                    channel_data = channel_data.sel({dim_name: dim_value})
                    channel_contrib = channel_contrib.sel({dim_name: dim_value})
                else:
                    channel_data = channel_data.sel({dim_name: dim_value})
                    channel_contrib = channel_contrib.sel({dim_name: dim_value})

        pc = azp.PlotCollection.grid(
            channel_contrib.mean(dim=["chain", "draw"]).to_dataset(),
            cols=additional_dims,
            rows=["channel"],
            aes={"color": ["channel"]},
            backend=backend,
        )
        pc.map(
            azp.visuals.scatter_xy,
            x=channel_data,
        )
        pc.map(azp.visuals.labelled_x, text="Channel Data", ignore_aes={"color"})
        pc.map(
            azp.visuals.labelled_y, text="Channel Contributions", ignore_aes={"color"}
        )
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        return pc

    def saturation_curves(
        self,
        curve: xr.DataArray,
        original_scale: bool = False,
        constant_data: xr.Dataset | None = None,
        posterior_data: xr.Dataset | None = None,
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Overlay saturation scatter plots with posterior predictive curves and HDI bands.

        Builds on saturation_scatterplot() by adding:
        - Sample curves from the posterior distribution
        - HDI bands showing uncertainty
        - Smooth saturation curves over the scatter plot

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        curve : xr.DataArray
            Posterior predictive saturation curves with required dimensions:
            - "chain", "draw": MCMC samples
            - "x": Input values for curve evaluation
            - "channel": Channel names

            Generate using: ``mmm.saturation.sample_curve(...)``
        original_scale : bool, default False
            Plot in original scale (True) or scaled space (False).
            If True, requires channel_contribution_original_scale in posterior.
        constant_data : xr.Dataset, optional
            Dataset containing constant_data group. If None, uses self.idata.constant_data.

            .. versionadded:: 0.18.0
               Added constant_data parameter for explicit data passing.

            This parameter allows testing with mock data and plotting alternative scenarios.
        posterior_data : xr.Dataset, optional
            Dataset containing posterior group. If None, uses self.idata.posterior.

            .. versionadded:: 0.18.0
               Added posterior_data parameter for explicit data passing.

            This parameter allows testing with mock posterior samples and comparing model fits.
        n_samples : int, default 10
            Number of sample curves to draw per subplot.
            Set to 0 to show only HDI bands without individual samples.
        hdi_probs : float or list of float, optional
            HDI probability levels for credible intervals.
            Examples: 0.94 (single band), [0.5, 0.94] (multiple bands).
            If None, no HDI bands are drawn.
        random_seed : np.random.Generator, optional
            Random number generator for reproducible curve sampling.
            If None, uses ``np.random.default_rng()``.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Examples:
            - {"geo": "US"}
            - {"geo": ["US", "UK"]}

            If provided, only the selected slice(s) will be plotted.
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default is "matplotlib".

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.

        Raises
        ------
        ValueError
            If curve is missing required dimensions ("x" or "channel").
        ValueError
            If original_scale=True but channel_contribution_original_scale not in posterior.

        See Also
        --------
        saturation_scatterplot : Base scatter plot without curves
        LegacyMMMPlotSuite.saturation_curves : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes)
        - Lost colors, subplot_kwargs, rc_params parameters
        - Different HDI calculation (uses arviz_plots instead of custom)

        Examples
        --------
        Generate and plot saturation curves:

        .. code-block:: python

            # Generate curves using saturation transformation
            curve = mmm.saturation.sample_curve(
                idata=mmm.idata.posterior[["saturation_beta", "saturation_lam"]],
                max_value=2.0,
            )
            pc = mmm.plot.saturation_curves(curve)
            pc.show()

        Add HDI bands:

        .. code-block:: python

            pc = mmm.plot.saturation_curves(curve, hdi_probs=[0.5, 0.94])
            pc.show()

        Original scale with custom seed:

        .. code-block:: python

            import numpy as np

            rng = np.random.default_rng(42)
            mmm.add_original_scale_contribution_variable(var=["channel_contribution"])
            pc = mmm.plot.saturation_curves(
                curve, original_scale=True, n_samples=15, random_seed=rng
            )
            pc.show()

        Filter by dimension:

        .. code-block:: python

            pc = mmm.plot.saturation_curves(curve, dims={"geo": "US"})
            pc.show()
        """
        # Get constant_data and posterior_data with fallback
        constant_data = self._get_data_or_fallback(
            constant_data, "constant_data", "constant data"
        )
        posterior_data = self._get_data_or_fallback(
            posterior_data, "posterior", "posterior data"
        )

        contrib_var = (
            "channel_contribution_original_scale"
            if original_scale
            else "channel_contribution"
        )

        if original_scale and contrib_var not in posterior_data:
            raise ValueError(
                f"""No posterior.{contrib_var} data found in posterior_data.\n"
                "Add a original scale deterministic:\n"
                "    mmm.add_original_scale_contribution_variable(\n"
                "        var=[\n"
                "            'channel_contribution',\n"
                "            ...\n"
                "        ]\n"
                "    )\n"
                """
            )
        # Validate curve dimensions
        if "x" not in curve.dims:
            raise ValueError("curve must have an 'x' dimension")
        if "channel" not in curve.dims:
            raise ValueError("curve must have a 'channel' dimension")

        if original_scale:
            curve_data = curve * constant_data.target_scale
            curve_data["x"] = curve_data["x"] * constant_data.channel_scale
        else:
            curve_data = curve
        curve_data = curve_data.rename("saturation_curve")

        # — 1. figure out grid shape based on scatter data dimensions / identify dims and combos
        cdims = constant_data.channel_data.dims
        all_dims = list(cdims)
        additional_dims = [d for d in cdims if d not in ("date", "channel")]
        # Validate dims and remove filtered dims from additional_dims
        if dims:
            self._validate_dims(dims, all_dims)
            additional_dims = [d for d in additional_dims if d not in dims]
        else:
            self._validate_dims({}, all_dims)

        # create the saturation scatterplot
        pc = self.saturation_scatterplot(
            original_scale=original_scale,
            constant_data=constant_data,
            posterior_data=posterior_data,
            dims=dims,
            backend=backend,
        )

        # add the hdi bands
        if hdi_probs is not None:
            # Robustly handle hdi_probs as float, list, tuple, or np.ndarray
            if isinstance(hdi_probs, (float, int)):
                hdi_probs_iter = [hdi_probs]
            elif isinstance(hdi_probs, (list, tuple, np.ndarray)):
                hdi_probs_iter = hdi_probs
            else:
                raise TypeError("hdi_probs must be a float, list, tuple, or np.ndarray")
            for hdi_prob in hdi_probs_iter:
                hdi = curve_data.azstats.hdi(hdi_prob)
                pc.map(
                    azp.visuals.fill_between_y,
                    x=curve_data["x"],
                    y_bottom=hdi.sel(ci_bound="lower"),
                    y_top=hdi.sel(ci_bound="upper"),
                    alpha=0.2,
                )

        if n_samples > 0:
            ##  sample the curves
            rng = np.random.default_rng(random_seed)

            # Stack the two dimensions
            stacked = curve_data.stack(sample=("chain", "draw"))

            # Sample from the stacked dimension
            idx = rng.choice(stacked.sizes["sample"], size=n_samples, replace=False)

            # Select and unstack
            sampled_curves = stacked.isel(sample=idx)

            # plot the sampled curves
            pc.map(
                azp.visuals.multiple_lines, x_dim="x", data=sampled_curves, alpha=0.2
            )

        return pc

    def budget_allocation_roas(
        self,
        samples: xr.Dataset,
        dims: dict[str, str | int | list] | None = None,
        dims_to_group_by: list[str] | str | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot ROI (Return on Ad Spend) distributions for budget allocation scenarios.

        Visualizes the posterior distribution of ROI for each channel given a budget
        allocation. Useful for comparing ROI across channels and understanding
        optimization trade-offs.

        .. versionadded:: 0.18.0
           New method in MMMPlotSuite v2. This is different from the legacy
           budget_allocation() method which showed bar charts.

        Parameters
        ----------
        samples : xr.Dataset
            Dataset from budget allocation optimization containing:
            - 'channel_contribution_original_scale': Channel contributions
            - 'allocation': Allocated budget per channel
            - 'channel' dimension

            Typically obtained from: ``mmm.allocate_budget_to_maximize_response(...)``
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Examples:
            - {"geo": "US"}
            - {"geo": ["US", "UK"]}

            If provided, only the selected slice(s) will be plotted.
        dims_to_group_by : list[str] | str | None, optional
            Dimension(s) to group by for overlaying distributions.
            When specified, all ROI distributions for each coordinate of that
            dimension will be plotted together for comparison.

            - None (default): Each distribution plotted separately
            - Single string: Group by that dimension (e.g., "geo")
            - List of strings: Group by multiple dimensions (e.g., ["geo", "segment"])
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.

        Raises
        ------
        ValueError
            If 'channel' dimension not found in samples.
        ValueError
            If required variables not found in samples.

        See Also
        --------
        LegacyMMMPlotSuite.budget_allocation : Legacy bar chart method (different purpose)

        Notes
        -----
        This method is NEW in MMMPlotSuite v2 and serves a different purpose
        than the legacy ``budget_allocation()`` method:

        - **New method** (this): Shows ROI distributions (KDE plots)
        - **Legacy method**: Shows bar charts comparing spend vs contributions

        To use the legacy method, set: ``mmm_config["plot.use_v2"] = False``

        Examples
        --------
        Basic usage with budget optimization results:

        .. code-block:: python

            allocation_results = mmm.allocate_budget_to_maximize_response(
                total_budget=100_000, budget_bounds={"lower": 0.5, "upper": 2.0}
            )
            pc = mmm.plot.budget_allocation_roas(allocation_results)
            pc.show()

        Group by geography to compare ROI across regions:

        .. code-block:: python

            pc = mmm.plot.budget_allocation_roas(
                allocation_results, dims_to_group_by="geo"
            )
            pc.show()

        Filter and group:

        .. code-block:: python

            pc = mmm.plot.budget_allocation_roas(
                allocation_results, dims={"segment": "premium"}, dims_to_group_by="geo"
            )
            pc.show()
        """
        # Get the channels from samples
        if "channel" not in samples.dims:
            raise ValueError(
                "Expected 'channel' dimension in samples dataset, but none found."
            )

        # Check for required variables in samples
        if "channel_contribution_original_scale" not in samples.data_vars:
            raise ValueError(
                "Expected a variable containing 'channel_contribution_original_scale' in samples, but none found."
            )
        if "allocation" not in samples:
            raise ValueError(
                "Expected 'allocation' variable in samples, but none found."
            )

        # Find the variable containing 'channel_contribution' in its name
        channel_contrib_var = "channel_contribution_original_scale"

        all_dims = list(samples.dims)
        # Validate dims
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)
        else:
            self._validate_dims({}, all_dims)

        channel_contribution = samples[channel_contrib_var].sum(dim="date")
        channel_contribution.name = "channel_contribution"

        from arviz_base import convert_to_datatree

        roa_da = channel_contribution / samples.allocation
        roa_dt = convert_to_datatree(roa_da)
        if isinstance(dims_to_group_by, str):
            dims_to_group_by = [dims_to_group_by]
        if dims_to_group_by:
            grouped = {"all": roa_dt.copy()}
            for dim in dims_to_group_by:
                new_grouped = {}
                for curr_k, curr_group in grouped.items():
                    curr_coords = curr_group.posterior.coords[dim].values
                    new_grouped.update(
                        {
                            f"{curr_k}, {dim}: {key}": curr_group.sel({dim: key})
                            for key in curr_coords
                        }
                    )
                grouped = new_grouped

            grouped_roa_dt = {}
            prefix = "all, "
            for k, v in grouped.items():
                if k.startswith(prefix):
                    grouped_roa_dt[k[len(prefix) :]] = v
                else:
                    grouped_roa_dt[k] = v
        else:
            grouped_roa_dt = roa_dt

        pc = azp.plot_dist(
            grouped_roa_dt,
            kind="kde",
            sample_dims=["sample"],
            backend=backend,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        )

        if dims_to_group_by:
            pc.add_legend(dim="model", title="")

        return pc

    def allocated_contribution_by_channel_over_time(
        self,
        samples: xr.Dataset,
        hdi_prob: float = 0.85,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot channel contributions over time from budget allocation optimization.

        Visualizes how contributions from each channel evolve over time given an
        optimized budget allocation. Shows mean contribution lines per channel with
        HDI uncertainty bands.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        samples : xr.Dataset
            Dataset from budget allocation optimization containing channel
            contributions over time. Required dimensions:
            - 'channel': Channel names
            - 'date': Time dimension
            - 'sample': MCMC samples

            Required variables:
            - Variable containing 'channel_contribution' (e.g., 'channel_contribution'
              or 'channel_contribution_original_scale')

            Typically obtained from: ``mmm.allocate_budget_to_maximize_response(...)``
        hdi_prob : float, default 0.85
            Probability mass for HDI interval (between 0 and 1).
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If required dimensions ('channel', 'date', 'sample') not found in samples.
        ValueError
            If no variable containing 'channel_contribution' found in samples.

        See Also
        --------
        budget_allocation_roas : Plot ROI distributions from same allocation results
        LegacyMMMPlotSuite.allocated_contribution_by_channel_over_time : Legacy implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes)
        - Lost scale_factor, lower_quantile, upper_quantile, figsize, ax parameters
        - Now uses HDI instead of quantiles for uncertainty
        - Automatic handling of extra dimensions (creates subplots)

        Examples
        --------
        Basic usage with budget optimization results:

        .. code-block:: python

            allocation_results = mmm.allocate_budget_to_maximize_response(
                total_budget=100_000, budget_bounds={"lower": 0.5, "upper": 2.0}
            )
            pc = mmm.plot.allocated_contribution_by_channel_over_time(
                allocation_results
            )
            pc.show()

        Custom HDI probability:

        .. code-block:: python

            pc = mmm.plot.allocated_contribution_by_channel_over_time(
                allocation_results, hdi_prob=0.94
            )
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.allocated_contribution_by_channel_over_time(
                allocation_results, backend="plotly"
            )
            pc.show()
        """
        # Check for expected dimensions and variables
        if "channel" not in samples.dims:
            raise ValueError(
                "Expected 'channel' dimension in samples dataset, but none found."
            )
        if "date" not in samples.dims:
            raise ValueError(
                "Expected 'date' dimension in samples dataset, but none found."
            )
        if "sample" not in samples.dims:
            raise ValueError(
                "Expected 'sample' dimension in samples dataset, but none found."
            )
        # Check if any variable contains channel contributions
        if not any(
            "channel_contribution" in var_name for var_name in samples.data_vars
        ):
            raise ValueError(
                "Expected a variable containing 'channel_contribution' in samples, but none found."
            )

        # Get channel contributions data
        channel_contrib_var = next(
            var_name
            for var_name in samples.data_vars
            if "channel_contribution" in var_name
        )

        # Identify extra dimensions beyond 'channel', 'date', and 'sample'
        all_dims = list(samples[channel_contrib_var].dims)
        ignored_dims = {"channel", "date", "sample"}
        extra_dims = [dim for dim in all_dims if dim not in ignored_dims]

        pc = azp.PlotCollection.wrap(
            samples[channel_contrib_var].to_dataset(),
            cols=extra_dims,
            aes={"color": ["channel"]},
            col_wrap=1,
            figure_kwargs={
                "sharex": True,
            },
            backend=backend,
        )

        # plot hdi
        hdi = samples[channel_contrib_var].azstats.hdi(hdi_prob, dim="sample")
        pc.map(
            azp.visuals.fill_between_y,
            x=samples[channel_contrib_var]["date"],
            y_bottom=hdi.sel(ci_bound="lower"),
            y_top=hdi.sel(ci_bound="upper"),
            alpha=0.2,
        )

        # plot mean contribution line
        pc.map(
            azp.visuals.line_xy,
            x=samples[channel_contrib_var]["date"],
            y=samples[channel_contrib_var].mean(dim="sample"),
        )

        pc.map(azp.visuals.labelled_x, text="Date", ignore_aes={"color"})
        pc.map(
            azp.visuals.labelled_y, text="Channel Contribution", ignore_aes={"color"}
        )
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
            ignore_aes={"color"},
        )

        pc.add_legend(dim="channel")
        return pc

    def _sensitivity_analysis_plot(
        self,
        data: xr.DataArray | xr.Dataset,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Private helper for plotting sensitivity analysis results.

        This is an internal method that performs the core plotting logic for
        sensitivity analysis visualizations. Public methods (sensitivity_analysis,
        uplift_curve, marginal_curve) handle data retrieval and call this helper.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset
            Sensitivity analysis data to plot. Must have required dimensions:
            - 'sample': MCMC samples
            - 'sweep': Sweep values (e.g., multipliers or input values)

            If Dataset, should contain 'x' variable.

            IMPORTANT: This parameter is REQUIRED with no fallback to self.idata.
            This design maintains separation of concerns - public methods handle
            data retrieval, this helper handles pure plotting.
        hdi_prob : float, default 0.94
            HDI probability mass (between 0 and 1).
        aggregation : dict, optional
            Aggregations to apply before plotting.
            Keys are operations ("sum", "mean", "median"), values are dimension tuples.
            Example: {"sum": ("channel",)} sums over the channel dimension.
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Note: Y-axis label is NOT set by this helper. Public methods calling
            this helper should set appropriate labels (e.g., "Contribution",
            "Uplift (%)", "Marginal Effect").

        Raises
        ------
        ValueError
            If data is missing required dimensions ('sample', 'sweep').

        Notes
        -----
        Design rationale for REQUIRED data parameter:

        - **Separation of concerns**: Public methods handle data location/retrieval
          (from self.idata.sensitivity_analysis, self.idata.posterior, etc.),
          this helper handles pure visualization logic.
        - **Testability**: Easy to test plotting logic with mock data.
        - **Cleaner implementation**: No monkey-patching or state manipulation.
        - **Flexibility**: Can be reused for different data sources without
          coupling to self.idata structure.

        This is a PRIVATE method (starts with _) and should not be called directly
        by users. Use public methods instead:
        - sensitivity_analysis(): General sensitivity analysis plots
        - uplift_curve(): Uplift percentage plots
        - marginal_curve(): Marginal effects plots
        """
        # Handle Dataset or DataArray
        x = data["x"] if isinstance(data, xr.Dataset) else data

        # Validate dimensions
        required_dims = {"sample", "sweep"}
        if not required_dims.issubset(set(x.dims)):
            raise ValueError(
                f"Data must have dimensions {required_dims}, got {set(x.dims)}"
            )
        # Coerce numeric dtype
        try:
            x = x.astype(float)
        except Exception as err:
            import warnings

            warnings.warn(
                f"Failed to cast sensitivity analysis data to float: {err}",
                RuntimeWarning,
                stacklevel=2,
            )
        # Apply aggregations
        if aggregation:
            for op, dims in aggregation.items():
                dims_list = [d for d in dims if d in x.dims]
                if not dims_list:
                    continue
                if op == "sum":
                    x = x.sum(dim=dims_list)
                elif op == "mean":
                    x = x.mean(dim=dims_list)
                else:
                    x = x.median(dim=dims_list)

        # Determine plotting dimensions (excluding sample & sweep)
        plot_dims = set(x.dims) - {"sample", "sweep"}

        pc = azp.PlotCollection.wrap(
            x.to_dataset(),
            cols=plot_dims,
            col_wrap=2,
            figure_kwargs={
                "sharex": True,
            },
            backend=backend,
        )

        # plot hdi
        hdi = x.azstats.hdi(hdi_prob, dim="sample")
        pc.map(
            azp.visuals.fill_between_y,
            x=x["sweep"],
            y_bottom=hdi.sel(ci_bound="lower"),
            y_top=hdi.sel(ci_bound="upper"),
            alpha=0.4,
            color="C0",
        )
        # plot aggregated line
        pc.map(
            azp.visuals.line_xy,
            x=x["sweep"],
            y=x.mean(dim="sample"),
            color="C0",
        )
        # add labels
        pc.map(azp.visuals.labelled_x, text="Sweep")
        pc.map(
            azp.visuals.labelled_title,
            subset_info=True,
            labeller=mix_labellers((NoVarLabeller, DimCoordLabeller))(),
        )
        return pc

    def sensitivity_analysis(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot sensitivity analysis results showing response to input changes.

        Visualizes how model outputs (e.g., channel contributions) change as inputs
        (e.g., channel spend) are varied. Shows mean response line and HDI bands
        across sweep values.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset, optional
            Sensitivity analysis data with required dimensions:
            - 'sample': MCMC samples
            - 'sweep': Sweep values (e.g., multipliers)

            If Dataset, should contain 'x' variable.
            If None, uses self.idata.sensitivity_analysis.

            .. versionadded:: 0.18.0
               Added data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock sensitivity analysis results
            - Plotting external sweep results
            - Comparing different sensitivity analyses
        hdi_prob : float, default 0.94
            HDI probability mass (between 0 and 1).
        aggregation : dict, optional
            Aggregations to apply before plotting.
            Keys: "sum", "mean", or "median"
            Values: tuple of dimension names

            Example: ``{"sum": ("channel",)}`` sums over channels before plotting.
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)`` or ``Axes``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If no sensitivity analysis data found in self.idata and no data provided.

        See Also
        --------
        uplift_curve : Plot uplift percentages (derived from sensitivity analysis)
        marginal_curve : Plot marginal effects (derived from sensitivity analysis)
        LegacyMMMPlotSuite.sensitivity_analysis : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes) or Axes
        - Lost ax, subplot_kwargs, plot_kwargs parameters (use backend methods)
        - Cleaner implementation without monkey-patching
        - Data parameter for explicit data passing (no side effects on self.idata)

        Examples
        --------
        Run sweep and plot results:

        .. code-block:: python

            from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

            # Run sensitivity sweep
            sweeps = np.linspace(0.5, 1.5, 11)
            sa = SensitivityAnalysis(mmm.model, mmm.idata)
            results = sa.run_sweep(
                var_input="channel_data",
                sweep_values=sweeps,
                var_names="channel_contribution",
                sweep_type="multiplicative",
                extend_idata=True,  # Store in idata
            )

            # Plot stored results
            pc = mmm.plot.sensitivity_analysis(hdi_prob=0.9)
            pc.show()

        Aggregate over channels:

        .. code-block:: python

            pc = mmm.plot.sensitivity_analysis(
                hdi_prob=0.9, aggregation={"sum": ("channel",)}
            )
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.sensitivity_analysis(backend="plotly")
            pc.show()

        Provide explicit data:

        .. code-block:: python

            external_results = sa.run_sweep(...)  # Not stored in idata
            pc = mmm.plot.sensitivity_analysis(data=external_results)
            pc.show()
        """
        # Retrieve data if not provided
        data = self._get_data_or_fallback(
            data, "sensitivity_analysis", "sensitivity analysis results"
        )

        pc = self._sensitivity_analysis_plot(
            data=data, hdi_prob=hdi_prob, aggregation=aggregation, backend=backend
        )
        pc.map(azp.visuals.labelled_y, text="Contribution")
        return pc

    def uplift_curve(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot uplift curves showing percentage change relative to baseline.

        Visualizes relative percentage changes in model outputs (e.g., channel
        contributions) as inputs are varied, compared to a reference point.
        Shows mean uplift line and HDI bands.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset, optional
            Uplift curve data computed from sensitivity analysis.
            If Dataset, should contain 'uplift_curve' variable.
            If None, uses self.idata.sensitivity_analysis['uplift_curve'].

            Must be precomputed using:
            ``SensitivityAnalysis.compute_uplift_curve_respect_to_base(...)``

            .. versionadded:: 0.18.0
               Added data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock uplift curve data
            - Plotting externally computed uplift curves
            - Comparing uplift curves from different models
        hdi_prob : float, default 0.94
            HDI probability mass (between 0 and 1).
        aggregation : dict, optional
            Aggregations to apply before plotting.
            Keys: "sum", "mean", or "median"
            Values: tuple of dimension names

            Example: ``{"sum": ("channel",)}`` sums over channels before plotting.
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)`` or ``Axes``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If no uplift curve data found in self.idata and no data provided.
        ValueError
            If 'uplift_curve' variable not found in sensitivity_analysis group.

        See Also
        --------
        sensitivity_analysis : Plot raw sensitivity analysis results
        marginal_curve : Plot marginal effects (absolute changes)
        LegacyMMMPlotSuite.uplift_curve : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes) or Axes
        - Cleaner implementation without monkey-patching
        - No longer modifies self.idata.sensitivity_analysis temporarily
        - Data parameter for explicit data passing

        Examples
        --------
        Compute and plot uplift curve:

        .. code-block:: python

            from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

            # Run sensitivity sweep
            sweeps = np.linspace(0.5, 1.5, 11)
            sa = SensitivityAnalysis(mmm.model, mmm.idata)
            results = sa.run_sweep(
                var_input="channel_data",
                sweep_values=sweeps,
                var_names="channel_contribution",
                sweep_type="multiplicative",
            )

            # Compute uplift relative to baseline (ref=1.0)
            uplift = sa.compute_uplift_curve_respect_to_base(
                results,
                ref=1.0,
                extend_idata=True,  # Store in idata
            )

            # Plot stored uplift curve
            pc = mmm.plot.uplift_curve(hdi_prob=0.9)
            pc.show()

        Aggregate over channels:

        .. code-block:: python

            pc = mmm.plot.uplift_curve(aggregation={"sum": ("channel",)})
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.uplift_curve(backend="plotly")
            pc.show()

        Provide explicit data:

        .. code-block:: python

            uplift_data = sa.compute_uplift_curve_respect_to_base(results, ref=1.0)
            pc = mmm.plot.uplift_curve(data=uplift_data)
            pc.show()
        """
        # Retrieve data if not provided
        if data is None:
            sa_group = self._get_data_or_fallback(
                None, "sensitivity_analysis", "sensitivity analysis results"
            )
            if isinstance(sa_group, xr.Dataset):
                if "uplift_curve" not in sa_group:
                    raise ValueError(
                        "Expected 'uplift_curve' in idata.sensitivity_analysis. "
                        "Use SensitivityAnalysis.compute_uplift_curve_respect_to_base(..., extend_idata=True)."
                    )
                data = sa_group["uplift_curve"]
            else:
                raise ValueError(
                    "sensitivity_analysis does not contain 'uplift_curve'. Did you persist it to idata?"
                )

        # Handle Dataset input
        if isinstance(data, xr.Dataset):
            if "uplift_curve" in data:
                data = data["uplift_curve"]
            elif "x" in data:
                data = data["x"]
            else:
                raise ValueError("Dataset must contain 'uplift_curve' or 'x' variable.")

        # Call helper with data (no more monkey-patching!)
        pc = self._sensitivity_analysis_plot(
            data=data,
            hdi_prob=hdi_prob,
            aggregation=aggregation,
            backend=backend,
        )
        pc.map(azp.visuals.labelled_y, text="Uplift (%)")
        return pc

    def marginal_curve(
        self,
        data: xr.DataArray | xr.Dataset | None = None,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot marginal effects showing absolute rate of change.

        Visualizes the instantaneous rate of change (derivative) of model outputs
        with respect to inputs. Shows how much output changes per unit change in
        input at each sweep value.

        .. versionadded:: 0.18.0
           New arviz_plots-based implementation supporting multiple backends.

        Parameters
        ----------
        data : xr.DataArray or xr.Dataset, optional
            Marginal effects data computed from sensitivity analysis.
            If Dataset, should contain 'marginal_effects' variable.
            If None, uses self.idata.sensitivity_analysis['marginal_effects'].

            Must be precomputed using:
            ``SensitivityAnalysis.compute_marginal_effects(...)``

            .. versionadded:: 0.18.0
               Added data parameter for explicit data passing.

            This parameter allows:
            - Testing with mock marginal effects data
            - Plotting externally computed marginal effects
            - Comparing marginal effects from different models
        hdi_prob : float, default 0.94
            HDI probability mass (between 0 and 1).
        aggregation : dict, optional
            Aggregations to apply before plotting.
            Keys: "sum", "mean", or "median"
            Values: tuple of dimension names

            Example: ``{"sum": ("channel",)}`` sums over channels before plotting.
        backend : str | None, optional
            Backend to use for plotting. If None, uses global backend configuration.

        Returns
        -------
        PlotCollection
            arviz_plots PlotCollection object containing the plot.

            Use ``.show()`` to display or ``.save("filename")`` to save.
            Unlike the legacy suite which returned ``(Figure, Axes)`` or ``Axes``,
            this provides a unified interface across all backends.

        Raises
        ------
        ValueError
            If no marginal effects data found in self.idata and no data provided.
        ValueError
            If 'marginal_effects' variable not found in sensitivity_analysis group.

        See Also
        --------
        sensitivity_analysis : Plot raw sensitivity analysis results
        uplift_curve : Plot uplift percentages (relative changes)
        LegacyMMMPlotSuite.marginal_curve : Legacy matplotlib-only implementation

        Notes
        -----
        Breaking changes from legacy implementation:

        - Returns PlotCollection instead of (Figure, Axes) or Axes
        - Cleaner implementation without monkey-patching
        - No longer modifies self.idata.sensitivity_analysis temporarily
        - Data parameter for explicit data passing

        Marginal effects show the **slope** of the sensitivity curve, helping
        identify where returns are diminishing most rapidly.

        Examples
        --------
        Compute and plot marginal effects:

        .. code-block:: python

            from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

            # Run sensitivity sweep
            sweeps = np.linspace(0.5, 1.5, 11)
            sa = SensitivityAnalysis(mmm.model, mmm.idata)
            results = sa.run_sweep(
                var_input="channel_data",
                sweep_values=sweeps,
                var_names="channel_contribution",
                sweep_type="multiplicative",
            )

            # Compute marginal effects (derivatives)
            me = sa.compute_marginal_effects(
                results,
                extend_idata=True,  # Store in idata
            )

            # Plot stored marginal effects
            pc = mmm.plot.marginal_curve(hdi_prob=0.9)
            pc.show()

        Aggregate over channels:

        .. code-block:: python

            pc = mmm.plot.marginal_curve(aggregation={"sum": ("channel",)})
            pc.show()

        Use different backend:

        .. code-block:: python

            pc = mmm.plot.marginal_curve(backend="plotly")
            pc.show()

        Provide explicit data:

        .. code-block:: python

            marginal_data = sa.compute_marginal_effects(results)
            pc = mmm.plot.marginal_curve(data=marginal_data)
            pc.show()
        """
        # Retrieve data if not provided
        if data is None:
            sa_group = self._get_data_or_fallback(
                None, "sensitivity_analysis", "sensitivity analysis results"
            )
            if isinstance(sa_group, xr.Dataset):
                if "marginal_effects" not in sa_group:
                    raise ValueError(
                        "Expected 'marginal_effects' in idata.sensitivity_analysis. "
                        "Use SensitivityAnalysis.compute_marginal_effects(..., extend_idata=True)."
                    )
                data = sa_group["marginal_effects"]
            else:
                raise ValueError(
                    "sensitivity_analysis does not contain 'marginal_effects'. Did you persist it to idata?"
                )

        # Handle Dataset input
        if isinstance(data, xr.Dataset):
            if "marginal_effects" in data:
                data = data["marginal_effects"]
            elif "x" in data:
                data = data["x"]
            else:
                raise ValueError(
                    "Dataset must contain 'marginal_effects' or 'x' variable."
                )

        # Call helper with data (no more monkey-patching!)
        pc = self._sensitivity_analysis_plot(
            data=data,
            hdi_prob=hdi_prob,
            aggregation=aggregation,
            backend=backend,
        )
        pc.map(azp.visuals.labelled_y, text="Marginal Effect")
        return pc

    def budget_allocation(self, *args, **kwargs):
        """
        Create bar chart comparing allocated spend and channel contributions.

        .. deprecated:: 0.18.0
           This method was removed in MMMPlotSuite v2. The arviz_plots library
           used in v2 doesn't support this specific chart type. See alternatives below.

        Raises
        ------
        NotImplementedError
            This method is not available in MMMPlotSuite v2.

        Notes
        -----
        Alternatives:

        1. **For ROI distributions**: Use :meth:`budget_allocation_roas`
           (different purpose but related to budget allocation)

        2. **To use the old method**: Switch to legacy suite:

           .. code-block:: python

               from pymc_marketing.mmm import mmm_config

               mmm_config["plot.use_v2"] = False
               mmm.plot.budget_allocation(samples)

        3. **Custom implementation**: Create bar chart using samples data:

           .. code-block:: python

               import matplotlib.pyplot as plt

               channel_contrib = samples["channel_contribution"].mean(...)
               allocated_spend = samples["allocation"]
               # Create custom bar chart with matplotlib

        See Also
        --------
        budget_allocation_roas : Plot ROI distributions by channel

        Examples
        --------
        Use legacy suite temporarily:

        .. code-block:: python

            from pymc_marketing.mmm import mmm_config

            original = mmm_config.get("plot.use_v2")
            try:
                mmm_config["plot.use_v2"] = False
                fig, ax = mmm.plot.budget_allocation(samples)
                fig.savefig("budget.png")
            finally:
                mmm_config["plot.use_v2"] = original
        """
        raise NotImplementedError(
            "budget_allocation() was removed in MMMPlotSuite v2.\n\n"
            "The new arviz_plots-based implementation doesn't support this chart type.\n\n"
            "Alternatives:\n"
            "  1. For ROI distributions: use budget_allocation_roas()\n"
            "  2. To use old method: set mmm_config['plot.use_v2'] = False\n"
            "  3. Implement custom bar chart using the samples data\n\n"
            "See documentation: https://docs.pymc-marketing.io/en/latest/mmm/plotting_migration.html#budget-allocation"
        )
