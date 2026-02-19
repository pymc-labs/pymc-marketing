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
    _ = mmm.plot.posterior_predictive(var=["y"], hdi_prob=0.9)

    # Residuals over time (true - predicted)
    _ = mmm.plot.residuals_over_time(hdi_prob=[0.94, 0.50])

    # Residuals posterior distribution
    _ = mmm.plot.residuals_posterior_distribution(aggregation="mean")

    # Posterior contributions over time (e.g., channel_contribution)
    _ = mmm.plot.contributions_over_time(var=["channel_contribution"], hdi_prob=0.9)

    # Posterior distribution of parameters (e.g., saturation parameter by channel)
    _ = mmm.plot.posterior_distribution(var="lam", plot_dim="channel")

    # Channel saturation scatter plot (scaled space by default)
    _ = mmm.plot.saturation_scatterplot(original_scale=False)

    # Channel contribution share forest plot
    _ = mmm.plot.channel_contribution_share_hdi(hdi_prob=0.94)

Wrap a custom PyMC model
--------

Requirements

- posterior_predictive plots: an `az.InferenceData` with a `posterior_predictive` group
  containing the variable(s) you want to plot with a `date` coordinate.
- residuals plots: a `posterior_predictive` group with `y_original_scale` variable (with `date`)
  and a `constant_data` group with `target_data` variable.
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
    _ = plot.posterior_predictive(var=["y"], hdi_prob=0.9)

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
import warnings
from collections.abc import Iterable
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pymc_marketing.metrics import crps
from pymc_marketing.mmm.utils import build_contributions

__all__ = ["MMMPlotSuite"]


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

    def _init_subplots(
        self,
        n_subplots: int,
        ncols: int = 1,
        width_per_col: float = 10.0,
        height_per_row: float = 4.0,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Initialize a grid of subplots.

        Parameters
        ----------
        n_subplots : int
            Number of rows (if ncols=1) or total subplots.
        ncols : int
            Number of columns in the subplot grid.
        width_per_col : float
            Width (in inches) for each column of subplots.
        height_per_row : float
            Height (in inches) for each row of subplots.
        figsize : tuple of float, optional
            If provided, overrides the calculated figure size (width, height) in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created Figure object.
        axes : np.ndarray of matplotlib.axes.Axes
            2D array of axes of shape (n_subplots, ncols).
        """
        if figsize is None:
            figsize = (width_per_col * ncols, height_per_row * n_subplots)
        fig, axes = plt.subplots(
            nrows=n_subplots,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
        )
        return fig, axes

    def _build_subplot_title(
        self,
        dims: list[str],
        combo: tuple,
        fallback_title: str = "Time Series",
    ) -> str:
        """Build a subplot title string from dimension names and their values."""
        if dims:
            title_parts = [f"{d}={v}" for d, v in zip(dims, combo, strict=False)]
            return ", ".join(title_parts)
        return fallback_title

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

    def _reduce_and_stack(
        self, data: xr.DataArray, dims_to_ignore: set[str] | None = None
    ) -> xr.DataArray:
        """Sum over leftover dims and stack chain+draw into sample if present."""
        if dims_to_ignore is None:
            dims_to_ignore = {"date", "chain", "draw", "sample"}

        leftover_dims = [d for d in data.dims if d not in dims_to_ignore]
        if leftover_dims:
            data = data.sum(dim=leftover_dims)

        # Combine chain+draw into 'sample' if both exist
        if "chain" in data.dims and "draw" in data.dims:
            data = data.stack(sample=("chain", "draw"))

        return data

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

    def _get_prior_predictive_data(
        self,
        idata: xr.Dataset | None,
    ) -> xr.Dataset:
        """Retrieve the prior_predictive group from either provided or self.idata."""
        if idata is not None:
            return idata

        # Otherwise, check if self.idata has posterior_predictive
        if (
            not hasattr(self.idata, "prior_predictive")  # type: ignore
            or self.idata.prior_predictive is None  # type: ignore
        ):
            raise ValueError(
                "No prior_predictive data found in 'self.idata'. "
                "Please run 'MMM.sample_prior_predictive()' or provide "
                "an external 'idata' argument."
            )
        return self.idata.prior_predictive  # type: ignore

    def _add_median_and_hdi(
        self,
        ax: Axes,
        data: xr.DataArray,
        var: str,
        hdi_prob: float = 0.85,
        label: str | None = None,
    ) -> Axes:
        """Add median and HDI to the given axis.

        Parameters
        ----------
        ax : Axes
            The matplotlib axes to plot on.
        data : xr.DataArray
            The data array containing samples.
        var : str
            The variable name (used as key for HDI results).
        hdi_prob : float, optional
            The HDI probability mass. Default is 0.85.
        label : str, optional
            The label to use in the legend. If None, uses `var`.

        Returns
        -------
        Axes
            The axes with the plot added.
        """
        if label is None:
            label = var
        median = data.median(dim="sample") if "sample" in data.dims else data.median()
        hdi = az.hdi(
            data,
            hdi_prob=hdi_prob,
            input_core_dims=[["sample"]] if "sample" in data.dims else None,
        )

        if "date" not in data.dims:
            raise ValueError(f"Expected 'date' dimension in {var}, but none found.")
        dates = data.coords["date"].values
        # Add median and HDI to the plot
        ax.plot(dates, median, label=label, alpha=0.9)
        ax.fill_between(dates, hdi[var][..., 0], hdi[var][..., 1], alpha=0.2)
        return ax

    def _add_gradient_to_axes(
        self,
        ax: Axes,
        data: xr.DataArray,
        n_percentiles: int = 30,
        palette: str = "Blues",
        **kwargs,
    ) -> Axes:
        """Add a gradient representation of the distribution to the axes.

        Creates a shaded area plot where color intensity represents
        the density of the distribution. Uses layered percentile ranges
        with varying opacity to create a smooth gradient effect.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to add the gradient to.
        data : xarray.DataArray
            The data array containing samples. Must have a 'sample' dimension
            and a dimension with coordinate values (typically 'date').
        n_percentiles : int, optional
            Number of percentile ranges to use for the gradient. More percentiles
            create a smoother gradient but increase rendering time. Default is 30.
        palette : str, optional
            Name of the matplotlib colormap to use. Default is "Blues".
        **kwargs
            Additional keyword arguments passed to ax.fill_between().

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the gradient added.

        Raises
        ------
        ValueError
            If data does not have a 'sample' dimension or lacks coordinate dimensions.
        """
        # Validate data has required dimensions
        if "sample" not in data.dims:
            raise ValueError(
                "Data must have a 'sample' dimension for gradient plotting."
            )

        # Find the coordinate dimension (typically 'date')
        coord_dims = [d for d in data.dims if d != "sample"]
        if not coord_dims:
            raise ValueError(
                "Data must have at least one coordinate dimension besides 'sample'."
            )
        coord_dim = coord_dims[0]  # Use first coordinate dimension
        x_values = data.coords[coord_dim].values

        # Set up color map and ranges
        cmap = plt.get_cmap(palette)
        color_range = np.linspace(0.3, 1.0, n_percentiles // 2)
        percentile_ranges = np.linspace(3, 97, n_percentiles)

        # Create gradient by filling between percentile ranges
        for i in range(len(percentile_ranges) - 1):
            # Compute percentiles along the sample dimension
            lower_percentile = np.percentile(
                data.values, percentile_ranges[i], axis=data.dims.index("sample")
            )
            upper_percentile = np.percentile(
                data.values, percentile_ranges[i + 1], axis=data.dims.index("sample")
            )

            # Map percentile index to color intensity
            # Middle percentiles get darker colors and higher alpha
            if i < n_percentiles // 2:
                color_val = color_range[i]
            else:
                color_val = color_range[n_percentiles - i - 2]

            # Alpha increases toward middle (50th percentile)
            alpha_val = 0.2 + 0.8 * (1 - abs(2 * i / n_percentiles - 1))

            ax.fill_between(
                x=x_values,
                y1=lower_percentile,
                y2=upper_percentile,
                color=cmap(color_val),
                alpha=alpha_val,
                **kwargs,
            )

        return ax

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

    def _filter_df_by_indexer(
        self, df: pd.DataFrame | None, indexer: dict
    ) -> pd.DataFrame:
        """Train / Test rows for this fold & panel: filter metadata DataFrames by all panel dims."""
        if df is None:
            return pd.DataFrame([])
        if not indexer:
            return df.copy()
        mask = pd.Series(True, index=df.index)
        for k, v in indexer.items():
            if k in df.columns:
                mask &= df[k].astype(str) == str(v)
        return df.loc[mask]

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self,
        var: list[str] | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
        add_gradient: bool = False,
        n_percentiles: int = 30,
        palette: str = "Blues",
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot time series from the posterior predictive distribution.

        By default, if both `var` and `idata` are not provided, uses
        `self.idata.posterior_predictive` and defaults the variable to `["y"]`.

        Parameters
        ----------
        var : list of str, optional
            A list of variable names to plot. Default is ["y"] if not provided.
        idata : xarray.Dataset, optional
            The posterior predictive dataset to plot. If not provided, tries to
            use `self.idata.posterior_predictive`.
        hdi_prob: float, optional
            The probability mass of the highest density interval to be displayed. Default is 0.85.
        add_gradient : bool, optional
            If True, add a gradient representation of the full distribution
            as a background layer. The gradient shows distribution density
            with color intensity. Default is False.
        n_percentiles : int, optional
            Number of percentile ranges to use for the gradient visualization.
            Only used when add_gradient=True. More percentiles create smoother
            gradients but increase rendering time. Default is 30.
        palette : str, optional
            Matplotlib colormap name for the gradient visualization.
            Only used when add_gradient=True. Common options: "Blues", "Reds",
            "Greens", "viridis", "plasma". Default is "Blues".

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.

        Raises
        ------
        ValueError
            If no `idata` is provided and `self.idata.posterior_predictive` does
            not exist, instructing the user to run `MMM.sample_posterior_predictive()`.
            If `hdi_prob` is not between 0 and 1, instructing the user to provide a valid value.

        Examples
        --------
        Basic usage with gradient:

        >>> fig, axes = mmm.plot.posterior_predictive(add_gradient=True)

        Customize gradient appearance:

        >>> fig, axes = mmm.plot.posterior_predictive(
        ...     add_gradient=True, n_percentiles=40, palette="viridis", hdi_prob=0.90
        ... )

        Combine gradient with HDI bands:

        >>> fig, axes = mmm.plot.posterior_predictive(add_gradient=True, hdi_prob=0.85)

        The gradient visualization shows distribution density where darker/more
        opaque colors indicate higher probability density (near the median) and
        lighter/more transparent colors indicate lower density (in the tails).

        Notes
        -----
        The gradient visualization uses a layered percentile approach where multiple
        percentile ranges are drawn as semi-transparent fills. The default uses 30
        percentile ranges from the 3rd to 97th percentile, creating a smooth gradient
        effect. Performance considerations:

        - More percentiles (higher n_percentiles) create smoother gradients but increase
          rendering time, especially with many subplots
        - The gradient is drawn as a background layer, with median and HDI overlaid on top
        - For multi-dimensional models, gradients are drawn independently for each subplot
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")
        # 1. Retrieve or validate posterior_predictive data
        pp_data = self._get_posterior_predictive_data(idata)

        # 2. Determine variables to plot
        if var is None:
            var = ["y"]
        main_var = var[0]

        # 3. Identify additional dims & get all combos
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, dim_combinations = self._get_additional_dim_combinations(
            data=pp_data, variable=main_var, ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        fig, axes = self._init_subplots(n_subplots=len(dim_combinations), ncols=1)

        # 5. Loop over dimension combinations
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # 6. Plot each requested variable
            for v in var:
                if v not in pp_data:
                    raise ValueError(
                        f"Variable '{v}' not in the posterior_predictive dataset."
                    )

                data = pp_data[v].sel(**indexers)
                # Sum leftover dims, stack chain+draw if needed
                data = self._reduce_and_stack(data, ignored_dims)

                # Add gradient visualization if requested (background layer)
                if add_gradient:
                    ax = self._add_gradient_to_axes(
                        ax=ax,
                        data=data,
                        n_percentiles=n_percentiles,
                        palette=palette,
                    )

                # Add median and HDI (foreground layer)
                ax = self._add_median_and_hdi(ax, data, v, hdi_prob=hdi_prob)

            # 7. Subplot title & labels
            title = self._build_subplot_title(
                dims=additional_dims,
                combo=combo,
                fallback_title="Posterior Predictive Time Series",
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Predictive")
            ax.legend(loc="best")

        return fig, axes

    def prior_predictive(
        self,
        var: str | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot time series from the posterior predictive distribution.

        By default, if both `var` and `idata` are not provided, uses
        `self.idata.posterior_predictive` and defaults the variable to `"y"`.

        Parameters
        ----------
        var : str, optional
            The variable name to plot. Default is "y" if not provided.
        idata : xarray.Dataset, optional
            The posterior predictive dataset to plot. If not provided, tries to
            use `self.idata.posterior_predictive`.
        hdi_prob: float, optional
            The probability mass of the highest density interval to be displayed. Default is 0.85.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.

        Raises
        ------
        ValueError
            If no `idata` is provided and `self.idata.posterior_predictive` does
            not exist, instructing the user to run `MMM.sample_posterior_predictive()`.
            If `hdi_prob` is not between 0 and 1, instructing the user to provide a valid value.
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")
        # 1. Retrieve or validate posterior_predictive data
        pp_data = self._get_prior_predictive_data(idata)

        # 2. Determine variable to plot
        if var is None:
            var = "y"
        main_var = var

        # 3. Identify additional dims & get all combos
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, dim_combinations = self._get_additional_dim_combinations(
            data=pp_data, variable=main_var, ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        fig, axes = self._init_subplots(n_subplots=len(dim_combinations), ncols=1)

        # 5. Loop over dimension combinations
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # 6. Plot the requested variable
            if var not in pp_data:
                raise ValueError(
                    f"Variable '{var}' not in the posterior_predictive dataset."
                )

            data = pp_data[var].sel(**indexers)
            # Sum leftover dims, stack chain+draw if needed
            data = self._reduce_and_stack(data, ignored_dims)
            ax = self._add_median_and_hdi(ax, data, var, hdi_prob=hdi_prob)

            # 7. Subplot title & labels
            title = self._build_subplot_title(
                dims=additional_dims,
                combo=combo,
                fallback_title="Posterior Predictive Time Series",
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Predictive")
            ax.legend(loc="best")

        return fig, axes

    def _compute_residuals(self) -> xr.DataArray:
        """Compute residuals (errors) as target - predictions.

        Returns
        -------
        xr.DataArray
            Residuals with name "residuals" and dimensions including chain, draw, date,
            and any additional model dimensions.

        Raises
        ------
        ValueError
            If `y_original_scale` is not in posterior_predictive.
            If `target_data` is not in constant_data.
        """
        # Check for required data
        pp_data = self._get_posterior_predictive_data(None)

        if "y_original_scale" not in pp_data:
            raise ValueError(
                "Variable 'y_original_scale' not found in posterior_predictive. "
                "This plot requires predictions in the original scale. "
                "Make sure to sample posterior_predictive after fitting the model."
            )

        if (
            not hasattr(self.idata, "constant_data")  # type: ignore
            or self.idata.constant_data is None  # type: ignore
            or "target_data" not in self.idata.constant_data  # type: ignore
        ):
            raise ValueError(
                "Variable 'target_data' not found in constant_data. "
                "This plot requires the target data to be stored in idata."
            )

        # Compute residuals
        target_data = self.idata.constant_data.target_data  # type: ignore
        predictions = pp_data["y_original_scale"]
        residuals = target_data - predictions
        residuals.name = "residuals"

        return residuals

    def residuals_over_time(
        self,
        hdi_prob: list[float] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot residuals over time by taking the difference between true values and predicted.

        Computes residuals = true values - predicted using target data from constant_data
        and predictions from posterior_predictive. Works with any model dimensionality.

        Parameters
        ----------
        hdi_prob : list of float, optional
            List of HDI probability masses to display. Default is [0.94].
            Each probability must be between 0 and 1. Multiple HDI bands will be
            plotted with decreasing transparency for wider bands.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.

        Raises
        ------
        ValueError
            If `y_original_scale` is not in posterior_predictive, instructing
            the user that this plot requires the original scale predictions.
            If `target_data` is not in constant_data.
            If any HDI probability is not between 0 and 1.

        Examples
        --------
        Plot residuals over time with default 94% HDI:

        .. code-block:: python

            mmm.plot.residuals_over_time()

        Plot residuals with multiple HDI bands:

        .. code-block:: python

            mmm.plot.residuals_over_time(hdi_prob=[0.94, 0.50])
        """
        # 1. Validate and set defaults
        if hdi_prob is None:
            hdi_prob = [0.94]

        for prob in hdi_prob:
            if not 0 < prob < 1:
                raise ValueError(
                    f"All HDI probabilities must be between 0 and 1, got {prob}."
                )

        # Sort probabilities in descending order (wider bands first)
        hdi_prob = sorted(hdi_prob, reverse=True)

        # 2. Compute residuals
        residuals = self._compute_residuals()
        pp_data = self._get_posterior_predictive_data(None)

        # 3. Identify additional dims & get all combos
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, dim_combinations = self._get_additional_dim_combinations(
            data=pp_data, variable="y_original_scale", ignored_dims=ignored_dims
        )

        # 4. Prepare subplots
        fig, axes = self._init_subplots(n_subplots=len(dim_combinations), ncols=1)

        # 5. Loop over dimension combinations
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # Select residuals for this combination
            residuals_subset = residuals.sel(**indexers)
            # Sum leftover dims, stack chain+draw if needed
            residuals_subset = self._reduce_and_stack(residuals_subset, ignored_dims)

            # Get date coordinate
            if "date" not in residuals_subset.dims:
                raise ValueError(
                    "Expected 'date' dimension in residuals, but none found."
                )
            dates = residuals_subset.coords["date"].values

            # 6. Plot HDI bands (wider bands first with lighter alpha)
            alphas = [0.2 + i * 0.2 for i in range(len(hdi_prob))]
            for prob, alpha in zip(hdi_prob, alphas, strict=True):
                residuals_hdi = az.hdi(
                    residuals_subset,
                    hdi_prob=prob,
                    input_core_dims=[["sample"]]
                    if "sample" in residuals_subset.dims
                    else None,
                )

                ax.fill_between(
                    dates,
                    residuals_hdi["residuals"].sel(hdi="lower"),
                    residuals_hdi["residuals"].sel(hdi="higher"),
                    color="C3",
                    alpha=alpha,
                    label=f"${100 * prob:.0f}\\%$ HDI",
                )

            # 7. Plot mean residual line
            mean_residuals = residuals_subset.mean(
                dim="sample" if "sample" in residuals_subset.dims else ("chain", "draw")
            )
            ax.plot(
                dates,
                mean_residuals.to_numpy(),
                color="C3",
                label="Residuals Mean",
            )

            # 8. Plot zero reference line
            ax.axhline(y=0.0, linestyle="--", color="black", label="zero")

            # 9. Subplot title & labels
            title = self._build_subplot_title(
                dims=additional_dims,
                combo=combo,
                fallback_title="Residuals Over Time",
            )
            ax.set_title(title)
            ax.set_xlabel("date")
            ax.set_ylabel("true - predictions")
            ax.legend(loc="best")

        return fig, axes

    def residuals_posterior_distribution(
        self,
        quantiles: list[float] | None = None,
        aggregation: str | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the posterior distribution of residuals.

        Displays the distribution of residuals (true - predicted) across all time points
        and dimensions. Users can choose to aggregate across dimensions using mean or sum.

        Parameters
        ----------
        quantiles : list of float, optional
            Quantiles to display on the distribution plot. Default is [0.25, 0.5, 0.75].
            Each value must be between 0 and 1.
        aggregation : str, optional
            How to aggregate residuals across non-chain/draw dimensions.
            Options: "mean", "sum", or None (default).
            - "mean": Average residuals across date and other dimensions
            - "sum": Sum residuals across date and other dimensions
            - None: Plot distribution for each dimension combination separately

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot.

        Raises
        ------
        ValueError
            If `y_original_scale` is not in posterior_predictive.
            If `target_data` is not in constant_data.
            If any quantile is not between 0 and 1.
            If aggregation is not one of "mean", "sum", or None.

        Examples
        --------
        Plot residuals distribution with default quantiles:

        .. code-block:: python

            mmm.plot.residuals_posterior_distribution()

        Plot with custom quantiles and aggregation:

        .. code-block:: python

            mmm.plot.residuals_posterior_distribution(
                quantiles=[0.05, 0.5, 0.95], aggregation="mean"
            )
        """
        # 1. Validate and set defaults
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]

        for q in quantiles:
            if not 0 <= q <= 1:
                raise ValueError(f"All quantiles must be between 0 and 1, got {q}.")

        if aggregation not in [None, "mean", "sum"]:
            raise ValueError(
                f"aggregation must be one of 'mean', 'sum', or None, got {aggregation!r}."
            )

        # 2. Compute residuals
        residuals = self._compute_residuals()
        pp_data = self._get_posterior_predictive_data(None)

        # 3. Handle aggregation
        if aggregation is not None:
            # Aggregate across all dimensions except chain and draw
            dims_to_agg = [d for d in residuals.dims if d not in ("chain", "draw")]
            if aggregation == "mean":
                residuals_agg = residuals.mean(dim=dims_to_agg)
            else:  # aggregation == "sum"
                residuals_agg = residuals.sum(dim=dims_to_agg)

            # Create single plot
            fig, ax = plt.subplots(figsize=(8, 6))
            az.plot_dist(
                residuals_agg,
                quantiles=quantiles,
                color="C3",
                fill_kwargs={"alpha": 0.7},
                ax=ax,
            )
            ax.axvline(x=0, color="black", linestyle="--", linewidth=1, label="zero")
            ax.legend()
            ax.set_title(f"Residuals Posterior Distribution ({aggregation})")
            ax.set_xlabel("Residuals")

            # Return as array for consistency
            axes = np.array([[ax]])
            return fig, axes

        # 4. Without aggregation: plot for each dimension combination
        ignored_dims = {"chain", "draw", "date", "sample"}
        additional_dims, dim_combinations = self._get_additional_dim_combinations(
            data=pp_data, variable="y_original_scale", ignored_dims=ignored_dims
        )

        # 5. Prepare subplots
        fig, axes = self._init_subplots(n_subplots=len(dim_combinations), ncols=1)

        # 6. Loop over dimension combinations
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # Select residuals for this combination and flatten over date
            residuals_subset = residuals.sel(**indexers)
            # Flatten date dimension for distribution plot
            if "date" in residuals_subset.dims:
                residuals_flat = residuals_subset.stack(
                    all_samples=("chain", "draw", "date")
                )
            else:
                residuals_flat = residuals_subset.stack(all_samples=("chain", "draw"))

            # Plot distribution
            az.plot_dist(
                residuals_flat,
                quantiles=quantiles,
                color="C3",
                fill_kwargs={"alpha": 0.7},
                ax=ax,
            )
            ax.axvline(x=0, color="black", linestyle="--", linewidth=1, label="zero")
            ax.legend()

            # Subplot title & labels
            title = self._build_subplot_title(
                dims=additional_dims,
                combo=combo,
                fallback_title="Residuals Posterior Distribution",
            )
            ax.set_title(title)
            ax.set_xlabel("Residuals")

        return fig, axes

    def contributions_over_time(
        self,
        var: list[str],
        hdi_prob: float = 0.85,
        dims: dict[str, str | int | list] | None = None,
        combine_dims: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the time-series contributions for each variable in `var`.

        showing the median and the credible interval (default 85%).
        Creates one subplot per combination of non-(chain/draw/date) dimensions
        and places all variables on the same subplot.

        Parameters
        ----------
        var : list of str
            A list of variable names to plot from the posterior.
        hdi_prob: float, optional
            The probability mass of the highest density interval to be displayed. Default is 0.85.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "user_type": "new"}.
            If provided, only the selected slice(s) will be plotted.
        combine_dims : bool, optional
            If True, all dimension combinations are plotted on a single axis with
            different colors. If False (default), creates separate subplots for each
            dimension combination.
        figsize : tuple of float, optional
            Figure size as (width, height) in inches. If None (default), size is
            calculated automatically based on the number of subplots.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.

        Raises
        ------
        ValueError
            If `hdi_prob` is not between 0 and 1, instructing the user to provide a valid value.
        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")

        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group."
            )

        main_var = var[0]
        all_dims = list(self.idata.posterior[main_var].dims)  # type: ignore
        ignored_dims = {"chain", "draw", "date"}
        additional_dims = [d for d in all_dims if d not in ignored_dims]

        coords = {
            key: value.to_numpy()
            for key, value in self.idata.posterior[var].coords.items()
        }

        # Apply user-specified filters (`dims`)
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)
            # Remove filtered dims from the combinations
            additional_dims = [d for d in additional_dims if d not in dims]
        else:
            self._validate_dims({}, all_dims)
            # additional_dims = [d for d in additional_dims if d not in dims]

        # Identify combos for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.posterior.coords[dim].values  # type: ignore
                for dim in additional_dims
            ]
            dim_combinations = list(itertools.product(*additional_coords))
        else:
            dim_combinations = [()]

        # If dims contains lists, build all combinations for those as well
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # Prepare subplots: one for each combo of dims_lists and additional_dims
        total_combos = list(itertools.product(dims_combos, dim_combinations))

        # --- Nested helper functions ---
        def build_indexers(dims_combo: tuple, addl_combo: tuple) -> dict[str, Any]:
            """Build indexers dictionary for a given dimension combination."""
            indexers: dict[str, Any] = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )
            if dims:
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                for k, v in dims.items():
                    if k not in dims_keys:
                        indexers[k] = v
            return indexers

        def prepare_var_data(
            var_name: str, indexers: dict[str, Any]
        ) -> tuple[xr.DataArray, dict[str, Any]]:
            """Prepare variable data for plotting."""
            data = self.idata.posterior[var_name]
            # Only expand 'date' if missing (needed for time series plotting)
            if "date" not in data.dims and "date" in coords:
                data = data.expand_dims(date=coords["date"])
            # Filter indexers to only include dimensions that exist in this variable
            var_indexers = {k: val for k, val in indexers.items() if k in data.dims}
            if var_indexers:
                data = data.sel(**var_indexers)
            data = self._reduce_and_stack(
                data, dims_to_ignore={"date", "chain", "draw", "sample"}
            )
            return data, var_indexers

        def get_title_dims() -> list[str]:
            """Get the list of dimensions for title/label building."""
            return list(dims.keys()) + additional_dims if dims else additional_dims

        # --- End nested helper functions ---

        title_dims = get_title_dims()

        if combine_dims:
            # Single subplot with all dimension combinations overlaid
            fig, axes = self._init_subplots(1, ncols=1, figsize=figsize)
            ax = axes[0][0]
            # Track variables without indexed dims to avoid duplicate plotting
            plotted_vars_without_indexed_dims: set[str] = set()

            for dims_combo, addl_combo in total_combos:
                indexers = build_indexers(dims_combo, addl_combo)
                title_combo = tuple(indexers[k] for k in title_dims)
                label_suffix = self._build_subplot_title(
                    dims=title_dims, combo=title_combo, fallback_title=""
                )

                for v in var:
                    data, var_indexers = prepare_var_data(v, indexers)
                    # Skip if this variable has no indexed dims and was already plotted
                    if not var_indexers:
                        if v in plotted_vars_without_indexed_dims:
                            continue
                        plotted_vars_without_indexed_dims.add(v)
                    # Create combined label: "var_name (dim=value, ...)"
                    if var_indexers and label_suffix:
                        plot_label = f"{v} ({label_suffix})"
                    else:
                        plot_label = v
                    ax = self._add_median_and_hdi(
                        ax, data, v, hdi_prob=hdi_prob, label=plot_label
                    )

            ax.set_title("Time Series Contributions")
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Value")
            ax.legend(loc="best")
        else:
            # Original behavior: separate subplots for each dimension combination
            fig, axes = self._init_subplots(len(total_combos), ncols=1, figsize=figsize)

            for row_idx, (dims_combo, addl_combo) in enumerate(total_combos):
                ax = axes[row_idx][0]
                indexers = build_indexers(dims_combo, addl_combo)

                for v in var:
                    data, _ = prepare_var_data(v, indexers)
                    ax = self._add_median_and_hdi(ax, data, v, hdi_prob=hdi_prob)

                title_combo = tuple(indexers[k] for k in title_dims)
                title = self._build_subplot_title(
                    dims=title_dims, combo=title_combo, fallback_title="Time Series"
                )
                ax.set_title(title)
                ax.set_xlabel("Date")
                ax.set_ylabel("Posterior Value")
                ax.legend(loc="best")

        return fig, axes

    def posterior_distribution(
        self,
        var: str,
        plot_dim: str = "channel",
        orient: str = "h",
        dims: dict[str, str | int | list] | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the posterior distribution of a variable across a specified dimension.

        Creates violin plots showing the posterior distribution of a parameter for each
        value in the specified dimension (e.g., each channel). If additional dimensions
        are present, creates a subplot for each combination.

        Parameters
        ----------
        var : str
            The name of the variable to plot from posterior.
        plot_dim : str, optional
            The dimension to plot distributions over. Default is "channel".
            This dimension will be used as the categorical axis for the violin plots.
        orient : str, optional
            Orientation of the plot. Either "h" (horizontal) or "v" (vertical).
            Default is "h".
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US", "channel": ["TV", "Radio"]}.
            If provided, only the selected slice(s) will be plotted.
        figsize : tuple[float, float], optional
            The size of each subplot. Default is (10, 6).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot.

        Raises
        ------
        ValueError
            If `var` is not found in the posterior.
            If `plot_dim` is not a dimension of the variable.
            If no posterior data is found in idata.

        Examples
        --------
        Plot posterior distribution of a saturation parameter:

        .. code-block:: python

            mmm.plot.posterior_distribution(var="lam", plot_dim="channel")

        Plot with dimension filtering:

        .. code-block:: python

            mmm.plot.posterior_distribution(
                var="lam", plot_dim="channel", dims={"geo": "US"}
            )

        Plot vertical orientation:

        .. code-block:: python

            mmm.plot.posterior_distribution(var="alpha", plot_dim="channel", orient="v")
        """
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group."
            )

        if var not in self.idata.posterior:
            raise ValueError(
                f"Variable '{var}' not found in posterior. "
                f"Available variables: {list(self.idata.posterior.data_vars)}"
            )

        var_data = self.idata.posterior[var]

        if plot_dim not in var_data.dims:
            raise ValueError(
                f"Dimension '{plot_dim}' not found in variable '{var}'. "
                f"Available dimensions: {list(var_data.dims)}"
            )

        all_dims = list(var_data.dims)

        # Validate dims parameter
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)
        else:
            self._validate_dims({}, all_dims)

        # Build all combinations for dims with lists
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # Identify additional dimensions (beyond chain, draw, and plot_dim)
        ignored_dims = {"chain", "draw", plot_dim}
        additional_dims = [
            d for d in all_dims if d not in ignored_dims and d not in (dims or {})
        ]

        # Get combinations for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.posterior.coords[dim].values for dim in additional_dims
            ]
            additional_combos = list(itertools.product(*additional_coords))
        else:
            additional_combos = [()]

        # Total combinations for subplots
        total_combos = list(itertools.product(dims_combos, additional_combos))
        n_subplots = len(total_combos)

        # Create subplots
        fig, axes = self._init_subplots(
            n_subplots=n_subplots,
            ncols=1,
            width_per_col=figsize[0],
            height_per_row=figsize[1],
        )

        for row_idx, (dims_combo, addl_combo) in enumerate(total_combos):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )

            if dims:
                # For dims with lists, use the current value from dims_combo
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                # For dims with single values, use as is
                for k, v in (dims or {}).items():
                    if k not in dims_keys:
                        indexers[k] = v

            # Select data for this subplot
            subset = var_data.sel(**indexers)

            # Extract samples and convert to DataFrame
            # Stack chain and draw into sample dimension
            if "chain" in subset.dims and "draw" in subset.dims:
                subset = subset.stack(sample=("chain", "draw"))

            # Get plot_dim values for labeling
            plot_dim_values = subset.coords[plot_dim].values

            # Convert to DataFrame for seaborn
            # Transpose so that plot_dim values are columns
            samples_df = pd.DataFrame(
                data=subset.values.T,
                columns=plot_dim_values,
            )

            # Create violin plot
            sns.violinplot(data=samples_df, orient=orient, ax=ax)

            # Build subplot title
            title_dims = (list(dims.keys()) if dims else []) + additional_dims
            title_combo = tuple(indexers[k] for k in title_dims)
            title = self._build_subplot_title(
                dims=title_dims,
                combo=title_combo,
                fallback_title=f"Posterior Distribution: {var}",
            )

            ax.set_title(title)

            if orient == "h":
                ax.set_xlabel(var)
                ax.set_ylabel(plot_dim)
            else:
                ax.set_xlabel(plot_dim)
                ax.set_ylabel(var)

        fig.tight_layout()
        return fig, axes

    def channel_parameter(
        self,
        param_name: str,
        orient: str = "h",
        dims: dict[str, str | int | list] | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Figure:
        """Plot the posterior distribution of a channel parameter using violin plots.

        Creates violin plots showing the posterior distribution of a parameter
        for each channel. Handles both channel-indexed parameters (with a "channel"
        dimension) and scalar parameters gracefully. If additional dimensions are
        present beyond chain, draw, and channel, creates a subplot for each combination.

        Parameters
        ----------
        param_name : str
            The name of the parameter to plot from posterior. Examples include
            "saturation_alpha", "saturation_lam", "adstock_alpha".
        orient : str, optional
            Orientation of the violin plot. Either "h" (horizontal) or "v" (vertical).
            Default is "h".
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US", "country": ["A", "B"]}.
            If provided, only the selected slice(s) will be plotted.
        figsize : tuple[float, float], optional
            The size of each subplot. Default is (10, 6).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.

        Raises
        ------
        ValueError
            If `param_name` is not found in the posterior.
            If no posterior data is found in idata.

        Examples
        --------
        Plot posterior distribution of saturation alpha parameter:

        .. code-block:: python

            fig = mmm.plot.channel_parameter(param_name="saturation_alpha")

        Plot with dimension filtering:

        .. code-block:: python

            fig = mmm.plot.channel_parameter(
                param_name="saturation_alpha", dims={"geo": "US"}
            )

        Plot with vertical orientation:

        .. code-block:: python

            fig = mmm.plot.channel_parameter(
                param_name="adstock_alpha", orient="v", figsize=(8, 10)
            )

        Add reference lines after plotting:

        .. code-block:: python

            fig = mmm.plot.channel_parameter(param_name="saturation_alpha")
            ax = fig.axes[0]
            ax.axvline(x=0.5, color="red", linestyle="--", label="reference")
            ax.legend()
        """
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group."
            )

        if param_name not in self.idata.posterior:
            raise ValueError(
                f"Parameter '{param_name}' not found in posterior. "
                f"Available variables: {list(self.idata.posterior.data_vars)}"
            )

        var_data = self.idata.posterior[param_name]
        all_dims = list(var_data.dims)

        # Determine if this is a channel-indexed parameter
        has_channel_dim = "channel" in all_dims

        # Validate dims parameter
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)

        # Build all combinations for dims with lists
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # Identify additional dimensions (beyond chain, draw, and channel if present)
        ignored_dims = {"chain", "draw"}
        if has_channel_dim:
            ignored_dims.add("channel")
        additional_dims = [
            d for d in all_dims if d not in ignored_dims and d not in (dims or {})
        ]

        # Get combinations for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.posterior.coords[dim].values for dim in additional_dims
            ]
            additional_combos = list(itertools.product(*additional_coords))
        else:
            additional_combos = [()]

        # Total combinations for subplots
        total_combos = list(itertools.product(dims_combos, additional_combos))
        n_subplots = len(total_combos)

        # Create subplots
        fig, axes = self._init_subplots(
            n_subplots=n_subplots,
            ncols=1,
            width_per_col=figsize[0],
            height_per_row=figsize[1],
        )

        for row_idx, (dims_combo, addl_combo) in enumerate(total_combos):
            ax = axes[row_idx][0]

            # Build indexers
            indexers = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )

            if dims:
                # For dims with lists, use the current value from dims_combo
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                # For dims with single values, use as is
                for k, v in (dims or {}).items():
                    if k not in dims_keys:
                        indexers[k] = v

            # Select data for this subplot
            subset = var_data.sel(**indexers)

            # Stack chain and draw into sample dimension
            if "chain" in subset.dims and "draw" in subset.dims:
                subset = subset.stack(sample=("chain", "draw"))

            if has_channel_dim:
                # Get channel values for labeling
                channel_values = subset.coords["channel"].values

                # Convert to DataFrame for seaborn
                # Transpose so that channel values are columns
                samples_df = pd.DataFrame(
                    data=subset.values.T,
                    columns=channel_values,
                )
            else:
                # Scalar parameter - create a single-column DataFrame
                samples_df = pd.DataFrame(
                    data=subset.values,
                    columns=[param_name],
                )

            # Create violin plot
            sns.violinplot(data=samples_df, orient=orient, ax=ax)

            # Build subplot title
            title_dims = (list(dims.keys()) if dims else []) + additional_dims
            title_combo = tuple(indexers[k] for k in title_dims)
            title = self._build_subplot_title(
                dims=title_dims,
                combo=title_combo,
                fallback_title=f"Posterior Distribution: {param_name}",
            )

            ax.set_title(title)

            if orient == "h":
                ax.set_xlabel(param_name)
                ax.set_ylabel("channel" if has_channel_dim else "")
            else:
                ax.set_xlabel("channel" if has_channel_dim else "")
                ax.set_ylabel(param_name)

        fig.tight_layout()
        return fig

    def prior_vs_posterior(
        self,
        var: str,
        plot_dim: str = "channel",
        alphabetical_sort: bool = True,
        dims: dict[str, str | int | list] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the prior vs posterior distribution for a variable across a dimension.

        Creates KDE plots showing the prior and posterior distributions with their
        respective means highlighted. Each subplot represents a value in the plot_dim
        (e.g., each channel). If additional dimensions are present, creates a grid
        of subplots for each combination.

        For scalar variables (those without the specified plot_dim), a single subplot
        is created showing the overall prior vs posterior comparison. If the variable
        has other dimensions besides chain/draw, subplots are created for each
        combination of those dimensions.

        Parameters
        ----------
        var : str
            The name of the variable to plot (e.g., 'adstock_alpha', 'lam').
        plot_dim : str, optional
            The dimension to create subplots over. Default is "channel".
            Each value in this dimension will get its own subplot showing
            prior vs posterior comparison. If the variable does not have this
            dimension, it is treated as a scalar variable.
        alphabetical_sort : bool, optional
            Whether to sort the plot_dim values alphabetically (True) or by the
            difference between the posterior and prior means (False), with the
            largest positive difference at the top. Default is True.
            Only applies when plot_dim exists in the variable.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"}.
            If provided, only the selected slice(s) will be plotted.
        figsize : tuple[float, float], optional
            The size of the figure. If None, it will be calculated based on
            the number of subplots.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot.

        Raises
        ------
        ValueError
            If `var` is not found in both prior and posterior.
            If no prior or posterior data is found in idata.

        Examples
        --------
        Plot prior vs posterior distribution of an adstock parameter:

        .. code-block:: python

            mmm.plot.prior_vs_posterior(var="adstock_alpha", plot_dim="channel")

        Plot a scalar variable (no channel dimension):

        .. code-block:: python

            mmm.plot.prior_vs_posterior(var="intercept")

        Plot with dimension filtering:

        .. code-block:: python

            mmm.plot.prior_vs_posterior(
                var="lam", plot_dim="channel", dims={"geo": "US"}
            )

        Sort by magnitude of update (largest posterior - prior difference first):

        .. code-block:: python

            mmm.plot.prior_vs_posterior(
                var="adstock_alpha", plot_dim="channel", alphabetical_sort=False
            )
        """
        # Validate that prior and posterior exist
        if not hasattr(self.idata, "prior") or self.idata.prior is None:
            raise ValueError(
                "No prior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'prior' group. "
                "Run 'MMM.sample_prior_predictive()' to generate prior samples."
            )

        if not hasattr(self.idata, "posterior") or self.idata.posterior is None:
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group. "
                "Run 'MMM.fit()' to generate posterior samples."
            )

        # Validate variable exists in both prior and posterior
        if var not in self.idata.prior:
            raise ValueError(
                f"Variable '{var}' not found in prior. "
                f"Available variables: {list(self.idata.prior.data_vars)}"
            )

        if var not in self.idata.posterior:
            raise ValueError(
                f"Variable '{var}' not found in posterior. "
                f"Available variables: {list(self.idata.posterior.data_vars)}"
            )

        prior_data = self.idata.prior[var]
        posterior_data = self.idata.posterior[var]

        all_dims = list(prior_data.dims)

        # Check if plot_dim exists - if not, treat as scalar variable
        is_scalar = plot_dim not in prior_data.dims

        if is_scalar:
            # Handle scalar variables (no plot_dim dimension)
            # Identify additional dimensions beyond chain, draw
            ignored_dims = {"chain", "draw"}
            additional_dims = [
                d for d in all_dims if d not in ignored_dims and d not in (dims or {})
            ]

            # Validate dims parameter if provided
            if dims:
                for key in dims:
                    if key not in all_dims:
                        raise ValueError(
                            f"Dimension '{key}' not found in variable '{var}'. "
                            f"Available dimensions: {list(prior_data.dims)}"
                        )

            # Get combinations for additional dims
            if additional_dims:
                additional_coords = [
                    self.idata.prior.coords[dim].values for dim in additional_dims
                ]
                additional_combos = list(itertools.product(*additional_coords))
            else:
                additional_combos = [()]

            # Calculate figsize if not provided
            if figsize is None:
                figsize = (12.0, 4.0)

            n_subplots = max(1, len(additional_combos))
            fig, axes = self._init_subplots(
                n_subplots=n_subplots,
                ncols=1,
                width_per_col=figsize[0],
                height_per_row=figsize[1],
            )

            # Plot for each additional dimension combination (or single plot if scalar)
            for row_idx, addl_combo in enumerate(additional_combos):
                ax = axes[row_idx][0]

                # Build indexers for additional dimensions
                indexers = (
                    dict(zip(additional_dims, addl_combo, strict=False))
                    if additional_dims
                    else {}
                )

                # Add single-value dims filters
                if dims:
                    for k, v in dims.items():
                        if not isinstance(v, (list, tuple, np.ndarray)):
                            indexers[k] = v

                # Extract samples
                prior_samples = prior_data.sel(**indexers).values.flatten()
                posterior_samples = posterior_data.sel(**indexers).values.flatten()
                prior_mean = float(np.mean(prior_samples))
                posterior_mean = float(np.mean(posterior_samples))
                difference = posterior_mean - prior_mean

                # Plot prior KDE
                sns.kdeplot(
                    prior_samples,
                    ax=ax,
                    label="Prior",
                    color="C0",
                    fill=True,
                )

                # Add vertical line for prior mean
                ax.axvline(
                    prior_mean,
                    color="C0",
                    linestyle="--",
                    linewidth=2,
                    label=f"Prior Mean: {prior_mean:.2f}",
                )

                # Plot posterior KDE
                sns.kdeplot(
                    posterior_samples,
                    ax=ax,
                    label="Posterior",
                    color="C1",
                    fill=True,
                    alpha=0.15,
                )

                # Add vertical line for posterior mean
                ax.axvline(
                    posterior_mean,
                    color="C1",
                    linestyle="--",
                    linewidth=2,
                    label=f"Posterior Mean: {posterior_mean:.2f} (Diff: {difference:.2f})",
                )

                # Build title
                if additional_dims:
                    title_parts = [
                        f"{d}={v}"
                        for d, v in zip(additional_dims, addl_combo, strict=False)
                    ]
                    ax.set_title(", ".join(title_parts))
                else:
                    ax.set_title(var)

                ax.set_xlabel(var)
                ax.set_ylabel("Density")
                ax.legend(loc="upper right", fontsize="small")

            fig.suptitle(
                f"Prior vs Posterior Distributions | {var}",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()
            return fig, axes

        # Non-scalar case: variable has the plot_dim dimension
        # Validate dims parameter
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)

        # Identify additional dimensions (beyond chain, draw, and plot_dim)
        ignored_dims = {"chain", "draw", plot_dim}
        additional_dims = [
            d for d in all_dims if d not in ignored_dims and d not in (dims or {})
        ]

        # Get combinations for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.prior.coords[dim].values for dim in additional_dims
            ]
            additional_combos = list(itertools.product(*additional_coords))
        else:
            additional_combos = [()]

        # Get plot_dim values
        plot_dim_values = prior_data.coords[plot_dim].values

        # Apply dims filter if provided for plot_dim
        if dims and plot_dim in dims:
            filter_val = dims[plot_dim]
            if isinstance(filter_val, (list, tuple, np.ndarray)):
                plot_dim_values = [v for v in plot_dim_values if v in filter_val]
            else:
                plot_dim_values = [filter_val]

        n_plot_dim = len(plot_dim_values)
        n_addl = len(additional_combos)

        # Calculate figsize if not provided
        if figsize is None:
            figsize = (12.0, 4.0)

        # Create subplots - one row per plot_dim value, one column per additional combo
        if n_addl > 1:
            fig, axes = self._init_subplots(
                n_subplots=n_plot_dim,
                ncols=n_addl,
                width_per_col=figsize[0] / max(n_addl, 1),
                height_per_row=figsize[1],
            )
        else:
            fig, axes = self._init_subplots(
                n_subplots=n_plot_dim,
                ncols=1,
                width_per_col=figsize[0],
                height_per_row=figsize[1],
            )

        # For each additional dimension combination, compute means and sort
        for addl_idx, addl_combo in enumerate(additional_combos):
            # Build indexers for additional dimensions
            addl_indexers = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )

            # Add single-value dims filters
            if dims:
                for k, v in dims.items():
                    if k not in [plot_dim] and not isinstance(
                        v, (list, tuple, np.ndarray)
                    ):
                        addl_indexers[k] = v

            # Compute prior and posterior means for sorting
            dim_means = []
            for dim_val in plot_dim_values:
                indexers = {**addl_indexers, plot_dim: dim_val}
                prior_samples = prior_data.sel(**indexers).values.flatten()
                posterior_samples = posterior_data.sel(**indexers).values.flatten()
                prior_mean = float(np.mean(prior_samples))
                posterior_mean = float(np.mean(posterior_samples))
                difference = posterior_mean - prior_mean
                dim_means.append((dim_val, prior_mean, posterior_mean, difference))

            # Sort based on alphabetical_sort parameter
            if alphabetical_sort:
                sorted_dims = sorted(dim_means, key=lambda x: str(x[0]))
            else:
                # Sort by difference (largest positive first)
                sorted_dims = sorted(dim_means, key=lambda x: x[3], reverse=True)

            # Plot for each plot_dim value
            for row_idx, (dim_val, prior_mean, posterior_mean, difference) in enumerate(
                sorted_dims
            ):
                ax = axes[row_idx][addl_idx]

                indexers = {**addl_indexers, plot_dim: dim_val}

                # Extract samples
                prior_samples = prior_data.sel(**indexers).values.flatten()
                posterior_samples = posterior_data.sel(**indexers).values.flatten()

                # Plot prior KDE
                sns.kdeplot(
                    prior_samples,
                    ax=ax,
                    label="Prior",
                    color="C0",
                    fill=True,
                )

                # Add vertical line for prior mean
                ax.axvline(
                    prior_mean,
                    color="C0",
                    linestyle="--",
                    linewidth=2,
                    label=f"Prior Mean: {prior_mean:.2f}",
                )

                # Plot posterior KDE
                sns.kdeplot(
                    posterior_samples,
                    ax=ax,
                    label="Posterior",
                    color="C1",
                    fill=True,
                    alpha=0.15,
                )

                # Add vertical line for posterior mean
                ax.axvline(
                    posterior_mean,
                    color="C1",
                    linestyle="--",
                    linewidth=2,
                    label=f"Posterior Mean: {posterior_mean:.2f} (Diff: {difference:.2f})",
                )

                # Build title
                title_parts = [f"{plot_dim}={dim_val}"]
                if additional_dims:
                    for d, v in zip(additional_dims, addl_combo, strict=False):
                        title_parts.append(f"{d}={v}")
                ax.set_title(", ".join(title_parts))
                ax.set_xlabel(var)
                ax.set_ylabel("Density")
                ax.legend(loc="upper right", fontsize="small")

        fig.suptitle(
            f"Prior vs Posterior Distributions | {var}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        return fig, axes

    def saturation_scatterplot(
        self,
        original_scale: bool = False,
        dims: dict[str, str | int | list] | None = None,
        **kwargs,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the saturation curves for each channel.

        Creates a grid of subplots for each combination of channel and non-(date/channel) dimensions.
        Optionally, subset by dims (single values or lists).
        Each channel will have a consistent color across all subplots.
        """
        if not hasattr(self.idata, "constant_data"):
            raise ValueError(
                "No 'constant_data' found in 'self.idata'. "
                "Please ensure 'self.idata' contains the constant_data group."
            )

        # Identify additional dimensions beyond 'date' and 'channel'
        cdims = self.idata.constant_data.channel_data.dims
        additional_dims = [dim for dim in cdims if dim not in ("date", "channel")]

        # Validate dims and remove filtered dims from additional_dims
        if dims:
            self._validate_dims(dims, list(self.idata.constant_data.channel_data.dims))
            additional_dims = [d for d in additional_dims if d not in dims]
        else:
            self._validate_dims({}, list(self.idata.constant_data.channel_data.dims))

        # Build all combinations for dims with lists
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # Build all combinations for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.constant_data.coords[d].values for d in additional_dims
            ]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        channels = self.idata.constant_data.coords["channel"].values
        n_channels = len(channels)
        n_addl = len(additional_combinations)
        n_dims = len(dims_combos)

        # For most use cases, n_dims will be 1, so grid is channels x additional_combinations
        # If dims_combos > 1, treat as extra axis (rare, but possible)
        nrows = n_channels
        ncols = n_addl * n_dims
        total_combos = list(
            itertools.product(channels, dims_combos, additional_combinations)
        )
        n_subplots = len(total_combos)

        # Assign a color to each channel
        channel_colors = {ch: f"C{i}" for i, ch in enumerate(channels)}

        # Prepare subplots as a grid
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                kwargs.get("width_per_col", 8) * ncols,
                kwargs.get("height_per_row", 4) * nrows,
            ),
            squeeze=False,
        )

        channel_contribution = (
            "channel_contribution_original_scale"
            if original_scale
            else "channel_contribution"
        )

        if original_scale and not hasattr(self.idata.posterior, channel_contribution):
            raise ValueError(
                f"""No posterior.{channel_contribution} data found in 'self.idata'. \n
                Add a original scale deterministic:\n
                mmm.add_original_scale_contribution_variable(\n
                var=[\n
                \"channel_contribution\",\n
                ...\n
                ]\n
                )\n
                """
            )

        for _idx, (channel, dims_combo, addl_combo) in enumerate(total_combos):
            # Compute subplot position
            row = list(channels).index(channel)
            # If dims_combos > 1, treat as extra axis (columns: addl * dims)
            if n_dims > 1:
                col = list(additional_combinations).index(addl_combo) * n_dims + list(
                    dims_combos
                ).index(dims_combo)
            else:
                col = list(additional_combinations).index(addl_combo)
            ax = axes[row][col]

            # Build indexers for dims and additional_dims
            indexers = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )
            if dims:
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                for k, v in (dims or {}).items():
                    if k not in dims_keys:
                        indexers[k] = v
            indexers["channel"] = channel

            # Select X data (constant_data)
            x_data = self.idata.constant_data.channel_data.sel(**indexers)
            # Select Y data (posterior contributions) and scale if needed
            y_data = self.idata.posterior[channel_contribution].sel(**indexers)
            y_data = y_data.mean(dim=[d for d in y_data.dims if d in ("chain", "draw")])
            x_data = x_data.broadcast_like(y_data)
            y_data = y_data.broadcast_like(x_data)
            ax.scatter(
                x_data.values.flatten(),
                y_data.values.flatten(),
                alpha=0.8,
                color=channel_colors[channel],
                label=str(channel),
            )
            # Build subplot title
            title_dims = (
                ["channel"] + (list(dims.keys()) if dims else []) + additional_dims
            )
            title_combo = (
                channel,
                *[indexers[k] for k in title_dims if k != "channel"],
            )
            title = self._build_subplot_title(
                dims=title_dims,
                combo=title_combo,
                fallback_title="Channel Saturation Curve",
            )
            ax.set_title(title)
            ax.set_xlabel("Channel Data (X)")
            ax.set_ylabel("Channel Contributions (Y)")
            ax.legend(loc="best")

        # Hide any unused axes (if grid is larger than needed)
        for i in range(nrows):
            for j in range(ncols):
                if i * ncols + j >= n_subplots:
                    axes[i][j].set_visible(False)

        return fig, axes

    def saturation_curves(
        self,
        curve: xr.DataArray,
        original_scale: bool = False,
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        colors: Iterable[str] | None = None,
        subplot_kwargs: dict | None = None,
        rc_params: dict | None = None,
        dims: dict[str, str | int | list] | None = None,
        **plot_kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Overlay saturation‑curve scatter‑plots with posterior‑predictive sample curves and HDI bands.

        **allowing** you to customize figsize and font sizes.

        Parameters
        ----------
        curve : xr.DataArray
            Posterior‑predictive curves (e.g. dims `("chain","draw","x","channel","geo")`).
        original_scale : bool, default=False
            Plot `channel_contribution_original_scale` if True, else `channel_contribution`.
        n_samples : int, default=10
            Number of sample‑curves per subplot.
        hdi_probs : float or list of float, optional
            Credible interval probabilities (e.g. 0.94 or [0.5, 0.94]).
            If None, uses ArviZ's default (0.94).
        random_seed : np.random.Generator, optional
            RNG for reproducible sampling. If None, uses `np.random.default_rng()`.
        colors : iterable of str, optional
            Colors for the sample & HDI plots.
        subplot_kwargs : dict, optional
            Passed to `plt.subplots` (e.g. `{"figsize": (10,8)}`).
            Merged with the function's own default sizing.
        rc_params : dict, optional
            Temporary `matplotlib.rcParams` for this plot.
            Example keys: `"xtick.labelsize"`, `"ytick.labelsize"`,
            `"axes.labelsize"`, `"axes.titlesize"`.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "region": "X"}.
            If provided, only the selected slice(s) will be plotted.
        **plot_kwargs
            Any other kwargs forwarded to `plot_curve`
            (for instance `same_axes=True`, `legend=True`, etc.).

        Returns
        -------
        fig : plt.Figure
            Matplotlib figure with your grid.
        axes : np.ndarray of plt.Axes
            Array of shape `(n_channels, n_geo)`.

        """
        from pymc_marketing.plot import plot_hdi, plot_samples

        if not hasattr(self.idata, "constant_data"):
            raise ValueError(
                "No 'constant_data' found in 'self.idata'. "
                "Please ensure 'self.idata' contains the constant_data group."
            )

        contrib_var = (
            "channel_contribution_original_scale"
            if original_scale
            else "channel_contribution"
        )

        if original_scale and not hasattr(self.idata.posterior, contrib_var):
            raise ValueError(
                f"""No posterior.{contrib_var} data found in 'self.idata'.\n"
                "Add a original scale deterministic:\n"
                "    mmm.add_original_scale_contribution_variable(\n"
                "        var=[\n"
                "            'channel_contribution',\n"
                "            ...\n"
                "        ]\n"
                "    )\n"
                """
            )
        curve_data = (
            curve * self.idata.constant_data.target_scale if original_scale else curve
        )
        curve_data = curve_data.rename("saturation_curve")

        # — 1. figure out grid shape based on scatter data dimensions / identify dims and combos
        cdims = self.idata.constant_data.channel_data.dims
        all_dims = list(cdims)
        additional_dims = [d for d in cdims if d not in ("date", "channel")]
        # Validate dims and remove filtered dims from additional_dims
        if dims:
            self._validate_dims(dims, all_dims)
            additional_dims = [d for d in additional_dims if d not in dims]
        else:
            self._validate_dims({}, all_dims)
        # Build all combinations for dims with lists
        dims_keys, dims_combos = self._dim_list_handler(dims)
        # Build all combinations for remaining dims
        if additional_dims:
            additional_coords = [
                self.idata.constant_data.coords[d].values for d in additional_dims
            ]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]
        channels = self.idata.constant_data.coords["channel"].values
        n_channels = len(channels)
        n_addl = len(additional_combinations)
        n_dims = len(dims_combos)
        nrows = n_channels
        ncols = n_addl * n_dims
        total_combos = list(
            itertools.product(channels, dims_combos, additional_combinations)
        )
        n_subplots = len(total_combos)

        # — 2. merge subplot_kwargs —
        user_subplot = subplot_kwargs or {}

        # Handle user-specified ncols/nrows
        if "ncols" in user_subplot:
            # User specified ncols, calculate nrows
            ncols = user_subplot["ncols"]
            nrows = int(np.ceil(n_subplots / ncols))
            user_subplot.pop("ncols")  # Remove to avoid conflict
        elif "nrows" in user_subplot:
            # User specified nrows, calculate ncols
            nrows = user_subplot["nrows"]
            ncols = int(np.ceil(n_subplots / nrows))
            user_subplot.pop("nrows")  # Remove to avoid conflict
        default_subplot = {"figsize": (ncols * 4, nrows * 3)}
        subkw = {**default_subplot, **user_subplot}
        # — 3. create subplots ourselves —
        rc_params = rc_params or {}
        with plt.rc_context(rc_params):
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subkw)
        # ensure a 2D array
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        if colors is None:
            colors = [f"C{i}" for i in range(n_channels)]
        elif not isinstance(colors, list):
            colors = list(colors)
        subplot_idx = 0
        for _idx, (ch, dims_combo, addl_combo) in enumerate(total_combos):
            if subplot_idx >= len(axes_flat):
                break
            ax = axes_flat[subplot_idx]
            subplot_idx += 1
            # Build indexers for dims and additional_dims
            indexers = (
                dict(zip(additional_dims, addl_combo, strict=False))
                if additional_dims
                else {}
            )
            if dims:
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                for k, v in (dims or {}).items():
                    if k not in dims_keys:
                        indexers[k] = v
            indexers["channel"] = ch
            # Select and broadcast curve data for this channel
            curve_idx = {
                dim: val for dim, val in indexers.items() if dim in curve_data.dims
            }
            subplot_curve = curve_data.sel(**curve_idx)
            if original_scale:
                valid_idx = {
                    k: v
                    for k, v in indexers.items()
                    if k in self.idata.constant_data.channel_scale.dims
                }
                channel_scale = self.idata.constant_data.channel_scale.sel(**valid_idx)
                x_original = subplot_curve.coords["x"] * channel_scale
                subplot_curve = subplot_curve.assign_coords(x=x_original)
            if n_samples > 0:
                plot_samples(
                    subplot_curve,
                    non_grid_names="x",
                    n=n_samples,
                    rng=random_seed,
                    axes=np.array([[ax]]),
                    colors=[colors[list(channels).index(ch)]],
                    same_axes=False,
                    legend=False,
                    **plot_kwargs,
                )
            if hdi_probs is not None:
                # Robustly handle hdi_probs as float, list, tuple, or np.ndarray
                if isinstance(hdi_probs, (float, int)):
                    hdi_probs_iter = [hdi_probs]
                elif isinstance(hdi_probs, (list, tuple, np.ndarray)):
                    hdi_probs_iter = hdi_probs
                else:
                    raise TypeError(
                        "hdi_probs must be a float, list, tuple, or np.ndarray"
                    )
                for hdi_prob in hdi_probs_iter:
                    plot_hdi(
                        subplot_curve,
                        non_grid_names="x",
                        hdi_prob=hdi_prob,
                        axes=np.array([[ax]]),
                        colors=[colors[list(channels).index(ch)]],
                        same_axes=False,
                        legend=False,
                        **plot_kwargs,
                    )
            x_data = self.idata.constant_data.channel_data.sel(**indexers)
            y = (
                self.idata.posterior[contrib_var]
                .sel(**indexers)
                .mean(
                    dim=[
                        d
                        for d in self.idata.posterior[contrib_var].dims
                        if d in ("chain", "draw")
                    ]
                )
            )
            x_data, y = x_data.broadcast_like(y), y.broadcast_like(x_data)
            ax.scatter(
                x_data.values.flatten(),
                y.values.flatten(),
                alpha=0.8,
                color=colors[list(channels).index(ch)],
            )
            title_dims = (
                ["channel"] + (list(dims.keys()) if dims else []) + additional_dims
            )
            title_combo = (
                ch,
                *[indexers[k] for k in title_dims if k != "channel"],
            )
            title = self._build_subplot_title(
                dims=title_dims,
                combo=title_combo,
                fallback_title="Channel Saturation Curves",
            )
            ax.set_title(title)
            ax.set_xlabel("Channel Data (X)")
            ax.set_ylabel("Channel Contribution (Y)")
        for ax_idx in range(subplot_idx, len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)
        return fig, axes

    def saturation_curves_scatter(
        self, original_scale: bool = False, **kwargs
    ) -> tuple[Figure, NDArray[Axes]]:
        """
        Plot scatter plots of channel contributions vs. channel data.

        .. deprecated:: 0.1.0
           Will be removed in version 0.2.0. Use :meth:`saturation_scatterplot` instead.

        Parameters
        ----------
        channel_contribution : str, optional
            Name of the channel contribution variable in the InferenceData.
        additional_dims : list[str], optional
            Additional dimensions to consider beyond 'channel'.
        additional_combinations : list[tuple], optional
            Specific combinations of additional dimensions to plot.
        **kwargs
            Additional keyword arguments passed to _init_subplots.

        Returns
        -------
        fig : plt.Figure
            The matplotlib figure.
        axes : np.ndarray
            Array of matplotlib axes.
        """
        import warnings

        warnings.warn(
            "saturation_curves_scatter is deprecated and will be removed in version 0.2.0. "
            "Use saturation_scatterplot instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Note: channel_contribution, additional_dims, and additional_combinations
        # are not used by saturation_scatterplot, so we don't pass them
        return self.saturation_scatterplot(original_scale=original_scale, **kwargs)

    def budget_allocation(
        self,
        samples: xr.Dataset,
        scale_factor: float | None = None,
        figsize: tuple[float, float] = (12, 6),
        ax: plt.Axes | None = None,
        original_scale: bool = True,
        dims: dict[str, str | int | list] | None = None,
    ) -> tuple[Figure, plt.Axes] | tuple[Figure, np.ndarray]:
        """Plot the budget allocation and channel contributions.

        Creates a bar chart comparing allocated spend and channel contributions
        for each channel. If additional dimensions besides 'channel' are present,
        creates a subplot for each combination of these dimensions.

        Parameters
        ----------
        samples : xr.Dataset
            The dataset containing the channel contributions and allocation values.
            Expected to have 'channel_contribution' and 'allocation' variables.
        scale_factor : float, optional
            Scale factor to convert to original scale, if original_scale=True.
            If None and original_scale=True, assumes scale_factor=1.
        figsize : tuple[float, float], optional
            The size of the figure to be created. Default is (12, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
            Only used when no extra dimensions are present.
        original_scale : bool, optional
            A boolean flag to determine if the values should be plotted in their
            original scale. Default is True.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "user_type": "new"}.
            If provided, only the selected slice(s) will be plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            The Axes object with the plot, or array of Axes for multiple subplots.
        """
        # Get the channels from samples
        if "channel" not in samples.dims:
            raise ValueError(
                "Expected 'channel' dimension in samples dataset, but none found."
            )

        # Check for required variables in samples
        if not any(
            "channel_contribution" in var_name for var_name in samples.data_vars
        ):
            raise ValueError(
                "Expected a variable containing 'channel_contribution' in samples, but none found."
            )
        if "allocation" not in samples:
            raise ValueError(
                "Expected 'allocation' variable in samples, but none found."
            )

        # Find the variable containing 'channel_contribution' in its name
        channel_contrib_var = next(
            var_name
            for var_name in samples.data_vars
            if "channel_contribution" in var_name
        )

        all_dims = list(samples.dims)
        # Validate dims
        if dims:
            self._validate_dims(dims=dims, all_dims=all_dims)
        else:
            self._validate_dims({}, all_dims)

        # Handle list-valued dims: build all combinations
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # After filtering with dims, only use extra dims not in dims and not ignored for subplotting
        ignored_dims = {"channel", "date", "sample", "chain", "draw"}
        channel_contribution_dims = list(samples[channel_contrib_var].dims)
        extra_dims = [
            d
            for d in channel_contribution_dims
            if d not in ignored_dims and d not in (dims or {})
        ]

        # Identify combos for remaining dims
        if extra_dims:
            extra_coords = [samples.coords[dim].values for dim in extra_dims]
            extra_combos = list(itertools.product(*extra_coords))
        else:
            extra_combos = [()]

        # Prepare subplots: one for each combo of dims_lists and extra_dims
        total_combos = list(itertools.product(dims_combos, extra_combos))
        n_subplots = len(total_combos)
        if n_subplots == 1 and ax is not None:
            axes = np.array([[ax]])
            fig = ax.get_figure()
        else:
            fig, axes = self._init_subplots(
                n_subplots=n_subplots,
                ncols=1,
                width_per_col=figsize[0],
                height_per_row=figsize[1],
            )

        for row_idx, (dims_combo, extra_combo) in enumerate(total_combos):
            ax_ = axes[row_idx][0]
            # Build indexers for dims and extra_dims
            indexers = (
                dict(zip(extra_dims, extra_combo, strict=False)) if extra_dims else {}
            )
            if dims:
                # For dims with lists, use the current value from dims_combo
                for i, k in enumerate(dims_keys):
                    indexers[k] = dims_combo[i]
                # For dims with single values, use as is
                for k, v in (dims or {}).items():
                    if k not in dims_keys:
                        indexers[k] = v

            # Select channel contributions for this subplot
            channel_contrib_data = samples[channel_contrib_var].sel(**indexers)
            allocation_data = samples.allocation
            # Only select dims that exist in allocation
            allocation_indexers = {
                k: v for k, v in indexers.items() if k in allocation_data.dims
            }
            allocation_data = allocation_data.sel(**allocation_indexers)

            # Average over all dims except channel (and those used for this subplot)
            used_dims = set(indexers.keys()) | {"channel"}
            reduction_dims = [
                dim for dim in channel_contrib_data.dims if dim not in used_dims
            ]
            channel_contribution = channel_contrib_data.mean(
                dim=reduction_dims
            ).to_numpy()
            if channel_contribution.ndim > 1:
                channel_contribution = channel_contribution.flatten()
            if original_scale and scale_factor is not None:
                channel_contribution *= scale_factor

            allocation_used_dims = set(allocation_indexers.keys()) | {"channel"}
            allocation_reduction_dims = [
                dim for dim in allocation_data.dims if dim not in allocation_used_dims
            ]
            if allocation_reduction_dims:
                allocated_spend = allocation_data.mean(
                    dim=allocation_reduction_dims
                ).to_numpy()
            else:
                allocated_spend = allocation_data.to_numpy()
            if allocated_spend.ndim > 1:
                allocated_spend = allocated_spend.flatten()

            self._plot_budget_allocation_bars(
                ax_,
                samples.coords["channel"].values,
                allocated_spend,
                channel_contribution,
            )

            # Build subplot title
            title_dims = (list(dims.keys()) if dims else []) + extra_dims
            title_combo = tuple(indexers[k] for k in title_dims)
            title = self._build_subplot_title(
                dims=title_dims,
                combo=title_combo,
                fallback_title="Budget Allocation",
            )
            ax_.set_title(title)

        fig.tight_layout()
        return fig, axes if n_subplots > 1 else (fig, axes[0][0])

    def _plot_budget_allocation_bars(
        self,
        ax: plt.Axes,
        channels: NDArray,
        allocated_spend: NDArray,
        channel_contribution: NDArray,
    ) -> None:
        """Plot budget allocation bars on a given axis.

        Parameters
        ----------
        ax : plt.Axes
            The axis to plot on.
        channels : NDArray
            Array of channel names.
        allocated_spend : NDArray
            Array of allocated spend values.
        channel_contribution : NDArray
            Array of channel contribution values.
        """
        bar_width = 0.35
        opacity = 0.7
        index = range(len(channels))

        # Plot allocated spend
        bars1 = ax.bar(
            index,
            allocated_spend,
            bar_width,
            color="C0",
            alpha=opacity,
            label="Allocated Spend",
        )

        # Create twin axis for contributions
        ax2 = ax.twinx()

        # Plot contributions
        bars2 = ax2.bar(
            [i + bar_width for i in index],
            channel_contribution,
            bar_width,
            color="C1",
            alpha=opacity,
            label="Channel Contribution",
        )

        # Labels and formatting
        ax.set_xlabel("Channels")
        ax.set_ylabel("Allocated Spend", color="C0", labelpad=10)
        ax2.set_ylabel("Channel Contributions", color="C1", labelpad=10)

        # Set x-ticks in the middle of the bars
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(channels)
        ax.tick_params(axis="x", rotation=90)

        # Turn off grid and add legend
        ax.grid(False)
        ax2.grid(False)

        bars = [bars1, bars2]
        labels = ["Allocated Spend", "Channel Contributions"]
        ax.legend(bars, labels, loc="best")

    def _prepare_allocated_contribution_data(
        self,
        samples: xr.Dataset | az.InferenceData,
        dims: dict[str, str | int | list] | None = None,
        split_by: str | list[str] | None = None,
        original_scale: bool = True,
        scale_factor: float | None = None,
    ) -> tuple[xr.DataArray, list[str], list[tuple]]:
        """Prepare channel contribution data with dimension filtering.

        This method handles validation, data extraction, dimension filtering,
        and split_by processing for the allocated contribution plot.

        Parameters
        ----------
        samples : xr.Dataset or az.InferenceData
            The dataset containing the samples of channel contributions.
            Can be an xr.Dataset with 'sample' dimension, or an az.InferenceData
            with 'chain' and 'draw' dimensions (from posterior_predictive).
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"}.
            If provided, only the selected slice(s) will be included.
        split_by : str or list of str, optional
            Dimension(s) to create separate subplots for. Each unique combination
            of values in these dimensions will get its own subplot.
        original_scale : bool, default True
            If True, prefer 'channel_contribution_original_scale' variable.
        scale_factor : float, optional
            Scale factor to apply to the contributions.

        Returns
        -------
        channel_contribution : xr.DataArray
            The channel contribution data, filtered and scaled, with 'sample' dim.
        split_dims : list of str
            List of dimension names to split by.
        dim_combinations : list of tuple
            List of coordinate value combinations for split dimensions.

        Raises
        ------
        ValueError
            If required dimensions or variables are missing.
        """
        # Handle InferenceData input - extract posterior_predictive
        if isinstance(samples, az.InferenceData):
            if hasattr(samples, "posterior_predictive"):
                samples = samples.posterior_predictive
            else:
                raise ValueError(
                    "InferenceData must contain 'posterior_predictive' group."
                )

        # Stack chain and draw into sample if present
        if "chain" in samples.dims and "draw" in samples.dims:
            samples = samples.stack(sample=("chain", "draw"))

        # Validate required dimensions
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
                "Expected a variable containing 'channel_contribution' in samples, "
                "but none found."
            )

        # Get channel contributions data - prefer original scale if requested
        if (
            original_scale
            and "channel_contribution_original_scale" in samples.data_vars
        ):
            channel_contrib_var = "channel_contribution_original_scale"
        else:
            channel_contrib_var = next(
                var_name
                for var_name in samples.data_vars
                if "channel_contribution" in var_name
            )

        channel_contribution = samples[channel_contrib_var]

        # Apply scale factor if provided
        if scale_factor is not None:
            channel_contribution = channel_contribution * scale_factor

        # Apply dimension filtering
        if dims is not None:
            selection = {}
            for key, val in dims.items():
                if key in channel_contribution.dims:
                    selection[key] = val
            if selection:
                channel_contribution = channel_contribution.sel(**selection)

        # Determine split dimensions and combinations
        if split_by is not None:
            split_dims = [split_by] if isinstance(split_by, str) else list(split_by)

            # Validate split dimensions exist in data
            for dim in split_dims:
                if dim not in channel_contribution.dims:
                    available_dims = list(channel_contribution.dims)
                    raise ValueError(
                        f"Split dimension '{dim}' not found in data dimensions. "
                        f"Available dimensions: {available_dims}"
                    )

            # Get unique combinations for split dimensions
            unique_values = [
                channel_contribution.coords[dim].values for dim in split_dims
            ]
            dim_combinations = list(itertools.product(*unique_values))
        else:
            # Auto-detect extra dimensions for backward compatibility
            ignored_dims = {"channel", "date", "sample"}
            extra_dims = [
                dim for dim in channel_contribution.dims if dim not in ignored_dims
            ]

            if extra_dims:
                split_dims = extra_dims
                unique_values = [
                    channel_contribution.coords[dim].values for dim in split_dims
                ]
                dim_combinations = list(itertools.product(*unique_values))
            else:
                split_dims = []
                dim_combinations = [()]

        return channel_contribution, split_dims, dim_combinations

    def _plot_single_allocated_contribution(
        self,
        ax: Axes,
        data: xr.DataArray,
        hdi_prob: float = 0.94,
        title: str = "Allocated Contribution by Channel Over Time",
    ) -> None:
        """Plot a single allocated contribution panel with HDI bands.

        This helper method renders a time series plot showing channel contributions
        over time with uncertainty intervals using HDI.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        data : xr.DataArray
            DataArray with dimensions including 'sample', 'date', and 'channel'.
        hdi_prob : float, default 0.94
            The probability mass for the HDI interval.
        title : str, default "Allocated Contribution by Channel Over Time"
            Title for the subplot.
        """
        # Plot mean values by channel
        data.mean(dim="sample").plot(hue="channel", ax=ax)

        # Add HDI intervals for each channel
        channels = data.coords["channel"].values
        dates = data.coords["date"].values

        for i, channel in enumerate(channels):
            channel_data = data.sel(channel=channel)

            # Compute HDI per date using arviz
            # Transpose to have samples in first dimension: (sample, date) -> (date, sample)
            # Then compute HDI for each date point
            n_dates = len(dates)
            hdi_lower = np.zeros(n_dates)
            hdi_upper = np.zeros(n_dates)

            for j, date in enumerate(dates):
                date_samples = channel_data.sel(date=date).values.flatten()
                hdi_result = az.hdi(date_samples, hdi_prob=hdi_prob)
                hdi_lower[j] = hdi_result[0]
                hdi_upper[j] = hdi_result[1]

            ax.fill_between(
                x=dates,
                y1=hdi_lower,
                y2=hdi_upper,
                alpha=0.15,
                color=f"C{i}",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Channel Contribution")
        ax.set_title(title)

    def allocated_contribution_by_channel_over_time(
        self,
        samples: xr.Dataset | az.InferenceData,
        hdi_prob: float = 0.94,
        dims: dict[str, str | int | list] | None = None,
        split_by: str | list[str] | None = None,
        original_scale: bool = True,
        scale_factor: float | None = None,
        figsize: tuple[float, float] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes] | tuple[Figure, NDArray[Axes]]:
        """Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along
        with HDI (Highest Density Interval) uncertainty bands. Supports dimension
        filtering via `dims` and creating separate subplots via `split_by`.

        Parameters
        ----------
        samples : xr.Dataset or az.InferenceData
            The dataset containing the samples of channel contributions.
            Can be an xr.Dataset with 'sample' dimension, or az.InferenceData
            (e.g., from sample_response_distribution) with 'chain' and 'draw' dims.
        hdi_prob : float, default 0.94
            The probability mass for the HDI interval.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"} to filter to a
            single geo, or {"geo": ["US", "UK"]} to include multiple values.
        split_by : str or list of str, optional
            Dimension(s) to create separate subplots for. Each unique combination
            of values in these dimensions will get its own subplot.
            If None, auto-detects extra dimensions beyond 'channel', 'date', 'sample'.
        original_scale : bool, default True
            If True, prefer 'channel_contribution_original_scale' variable if available.
        scale_factor : float, optional
            Scale factor to apply to the contributions.
        figsize : tuple[float, float], optional
            The size of the figure. Default is (10, 6) for single panel,
            scaled automatically for multiple panels.
        subplot_kwargs : dict, optional
            Additional keyword arguments for subplot layout. Can include:
            - 'nrows': Number of rows in subplot grid
            - 'ncols': Number of columns in subplot grid
            Only one of 'nrows' or 'ncols' can be specified when using split_by.
        **kwargs
            Additional keyword arguments passed to plt.subplots().

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            The Axes object with the plot for single panel, or array of Axes
            for multiple subplots.

        Examples
        --------
        Basic usage with optimization samples:

        >>> allocation, _ = mmm.optimize_budget(budget=100_000, num_periods=52)
        >>> samples = mmm.sample_response_distribution(
        ...     allocation_strategy=allocation,
        ...     time_granularity="weekly",
        ...     num_periods=52,
        ...     noise_level=0.1,
        ... )
        >>> fig, ax = mmm.plot.allocated_contribution_by_channel_over_time(samples)

        Filter to a specific dimension value:

        >>> fig, ax = mmm.plot.allocated_contribution_by_channel_over_time(
        ...     samples, dims={"geo": "US"}
        ... )

        Create subplots split by dimension:

        >>> fig, axes = mmm.plot.allocated_contribution_by_channel_over_time(
        ...     samples, split_by="geo"
        ... )
        """
        # Handle subplot_kwargs
        if subplot_kwargs is None:
            subplot_kwargs = {}
        else:
            subplot_kwargs = subplot_kwargs.copy()

        # Validate nrows/ncols not both specified
        if "nrows" in subplot_kwargs and "ncols" in subplot_kwargs:
            raise ValueError(
                "Specify only one of 'nrows' or 'ncols' in subplot_kwargs, not both."
            )

        # Prepare data with dimension filtering and split_by handling
        channel_contribution, split_dims, dim_combinations = (
            self._prepare_allocated_contribution_data(
                samples=samples,
                dims=dims,
                split_by=split_by,
                original_scale=original_scale,
                scale_factor=scale_factor,
            )
        )

        n_panels = len(dim_combinations)

        # Single panel case
        if n_panels == 1:
            if figsize is None:
                figsize = (10, 6)

            fig, ax = plt.subplots(figsize=figsize, **kwargs)

            self._plot_single_allocated_contribution(
                ax=ax,
                data=channel_contribution,
                hdi_prob=hdi_prob,
                title="Allocated Contribution by Channel Over Time",
            )

            fig.tight_layout()
            return fig, ax

        # Multiple panels case
        if figsize is None:
            figsize = (10, 6)

        # Determine grid layout
        if "nrows" in subplot_kwargs:
            nrows = subplot_kwargs.pop("nrows")
            ncols = int(np.ceil(n_panels / nrows))
        elif "ncols" in subplot_kwargs:
            ncols = subplot_kwargs.pop("ncols")
            nrows = int(np.ceil(n_panels / ncols))
        else:
            # Default: single column layout
            nrows = n_panels
            ncols = 1

        # Calculate figure size based on grid
        subplot_figsize = (figsize[0] * ncols, figsize[1] * nrows)

        fig, axes_grid = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=subplot_figsize,
            layout="constrained",
            **subplot_kwargs,
            **kwargs,
        )

        # Normalize axes to 2D array
        if isinstance(axes_grid, Axes):
            axes_array = np.array([[axes_grid]])
        elif axes_grid.ndim == 1:
            axes_array = (
                axes_grid.reshape(1, -1) if nrows == 1 else axes_grid.reshape(-1, 1)
            )
        else:
            axes_array = axes_grid

        # Flatten for iteration
        axes_flat = axes_array.flatten()

        # Plot each combination
        for idx, combo in enumerate(dim_combinations):
            current_ax = axes_flat[idx]

            # Filter data for this combination
            if split_dims and combo:
                selection = dict(zip(split_dims, combo, strict=False))
                subset = channel_contribution.sel(**selection)
            else:
                subset = channel_contribution

            # Build subplot title
            title = self._build_subplot_title(
                dims=split_dims,
                combo=combo,
                fallback_title="Allocated Contribution by Channel Over Time",
            )

            self._plot_single_allocated_contribution(
                ax=current_ax,
                data=subset,
                hdi_prob=hdi_prob,
                title=title,
            )

        # Hide unused axes
        for ax_extra in axes_flat[n_panels:]:
            ax_extra.set_visible(False)

        return fig, axes_array

    def sensitivity_analysis(
        self,
        hdi_prob: float = 0.94,
        ax: plt.Axes | None = None,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        *,
        plot_kwargs: dict[str, Any] | None = None,
        ylabel: str = "Effect",
        xlabel: str = "Sweep",
        title: str | None = None,
        add_figure_title: bool = False,
        subplot_title_fallback: str = "Sensitivity Analysis",
        hue_dim: str | None = None,
        legend: bool | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        x_sweep_axis: Literal["relative", "absolute"] = "relative",
    ) -> tuple[Figure, NDArray[Axes]] | plt.Axes:
        """Plot sensitivity analysis results.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        ax : plt.Axes, optional
            The axis to plot on.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.

        Other Parameters
        ----------------
        plot_kwargs : dict, optional
            Keyword arguments forwarded to the underlying line plot. Defaults include
            ``{"color": "C0"}``.
        ylabel : str, optional
            Y-axis label. Defaults to "Effect".
        xlabel : str, optional
            X-axis label. Defaults to "Sweep".
        title : str, optional
            Figure-level title to add when ``add_figure_title=True``.
        add_figure_title : bool, optional
            Whether to add a figure-level title. Defaults to ``False``.
        subplot_title_fallback : str, optional
            Fallback title used for subplot titles when no plotting dims exist. Defaults
            to "Sensitivity Analysis".
        hue_dim : str, optional
            Dimension to draw multiple lines per subplot (e.g., "channel"). When provided,
            this dimension is excluded from the subplot grid.
        legend : bool, optional
            Whether to show a legend when ``hue_dim`` is provided. Defaults to ``True``
            when ``hue_dim`` is set.
        legend_kwargs : dict, optional
            Keyword arguments forwarded to ``Axes.legend`` when ``hue_dim`` is set.
        x_sweep_axis : {"relative", "absolute"}, optional
            Controls how the X-axis values are displayed. Defaults to ``"relative"``.

            - ``"relative"``: Shows sweep multipliers (e.g., 0.5x, 1.0x, 2.0x).
            - ``"absolute"``: Shows absolute spend values by multiplying sweep values
              by the ``channel_scale`` from ``idata.constant_data``. Requires ``hue_dim``
              to be set so each line can be scaled appropriately. Each channel will have
              its own X-axis range based on its scale factor.

        Examples
        --------
        Basic run using stored results in `idata`:

        .. code-block:: python

            # Assuming you already ran a sweep and stored results
            # under idata.sensitivity_analysis via SensitivityAnalysis.run_sweep(..., extend_idata=True)
            ax = mmm.plot.sensitivity_analysis(hdi_prob=0.9)

        With aggregation over dimensions (e.g., sum over channels):

        .. code-block:: python

            ax = mmm.plot.sensitivity_analysis(
                hdi_prob=0.9,
                aggregation={"sum": ("channel",)},
            )

        With multiple lines per subplot (e.g., channels within each geo):

        .. code-block:: python

            fig, axes = mmm.plot.sensitivity_analysis(
                hdi_prob=0.9,
                hue_dim="channel",
                subplot_kwargs={"nrows": 2, "figsize": (12, 8)},
            )

        With absolute X-axis values (requires ``hue_dim`` to be set):

        .. code-block:: python

            fig, axes = mmm.plot.sensitivity_analysis(
                hdi_prob=0.9,
                hue_dim="channel",
                x_sweep_axis="absolute",
                xlabel="Total Spend",
            )
        """
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found. Call .sensitivity.run_sweep() first."
            )
        sa = self.idata.sensitivity_analysis  # type: ignore
        x = sa["x"] if isinstance(sa, xr.Dataset) else sa
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
        # Determine plotting dimensions (excluding sample, sweep, and hue)
        excluded_dims = {"sample", "sweep"}
        if hue_dim is not None:
            if hue_dim in excluded_dims:
                raise ValueError(
                    f"Invalid hue_dim '{hue_dim}'. It cannot be one of {sorted(excluded_dims)}."
                )
            if hue_dim not in x.dims:
                raise ValueError(
                    f"Dimension '{hue_dim}' not found in sensitivity analysis results."
                )
            excluded_dims.add(hue_dim)

        # Validate and prepare for absolute x-axis
        channel_scale = None
        if x_sweep_axis == "absolute":
            if hue_dim is None:
                raise ValueError(
                    "x_sweep_axis='absolute' requires hue_dim to be set "
                    "(e.g., hue_dim='channel') so each line can be scaled appropriately."
                )
            if not hasattr(self.idata, "constant_data"):
                raise ValueError(
                    "x_sweep_axis='absolute' requires idata.constant_data to exist."
                )
            if "channel_scale" not in self.idata.constant_data:
                raise ValueError(
                    "x_sweep_axis='absolute' requires 'channel_scale' in "
                    "idata.constant_data."
                )
            channel_scale = self.idata.constant_data.channel_data.sum(dim="date")

        plot_dims = [d for d in x.dims if d not in excluded_dims]
        if plot_dims:
            dim_combinations = list(
                itertools.product(*[x.coords[d].values for d in plot_dims])
            )
        else:
            dim_combinations = [()]

        n_panels = len(dim_combinations)

        # Handle axis/grid creation
        subplot_kwargs = {**(subplot_kwargs or {})}
        nrows_user = subplot_kwargs.pop("nrows", None)
        ncols_user = subplot_kwargs.pop("ncols", None)
        if nrows_user is not None and ncols_user is not None:
            raise ValueError(
                "Specify only one of 'nrows' or 'ncols' in subplot_kwargs."
            )

        if n_panels > 1:
            if ax is not None:
                raise ValueError(
                    "Multiple sensitivity panels detected; please omit 'ax' and use 'subplot_kwargs' instead."
                )
            if ncols_user is not None:
                ncols = ncols_user
                nrows = int(np.ceil(n_panels / ncols))
            elif nrows_user is not None:
                nrows = nrows_user
                ncols = int(np.ceil(n_panels / nrows))
            else:
                ncols = max(1, int(np.ceil(np.sqrt(n_panels))))
                nrows = int(np.ceil(n_panels / ncols))
            subplot_kwargs.setdefault("figsize", (ncols * 4.0, nrows * 3.0))
            fig, axes_grid = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                **subplot_kwargs,
            )
            if isinstance(axes_grid, plt.Axes):
                axes_grid = np.array([[axes_grid]])
            elif axes_grid.ndim == 1:
                axes_grid = axes_grid.reshape(1, -1)
            axes_array = axes_grid
        else:
            if ax is not None:
                axes_array = np.array([[ax]])
                fig = ax.figure
            else:
                if ncols_user is not None or nrows_user is not None:
                    subplot_kwargs.setdefault("figsize", (4.0, 3.0))
                    fig, single_ax = plt.subplots(
                        nrows=1,
                        ncols=1,
                        **subplot_kwargs,
                    )
                else:
                    fig, single_ax = plt.subplots()
                axes_array = np.array([[single_ax]])

        # Merge plotting kwargs with defaults
        _plot_kwargs = {"color": "C0"}
        if plot_kwargs:
            _plot_kwargs.update(plot_kwargs)
        _line_color = _plot_kwargs.get("color", "C0")
        legend_on = legend if legend is not None else hue_dim is not None
        _legend_kwargs = legend_kwargs or {}
        hue_values: list[Any] = []
        if hue_dim is not None:
            hue_values = (
                list(x.coords[hue_dim].values)
                if hue_dim in x.coords
                else list(range(x.sizes[hue_dim]))
            )
            hue_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
            color_cycle = itertools.cycle(hue_colors)

        def _plot_line(
            line_data: xr.DataArray,
            line_kwargs: dict[str, Any],
            line_color: str,
            x_override: np.ndarray | None = None,
        ) -> None:
            line_data = line_data.squeeze(drop=True).astype(float)

            if "sweep" in line_data.dims:
                sweep_dim = "sweep"
            else:
                cand = [d for d in line_data.dims if d != "sample"]
                if not cand:
                    raise ValueError(
                        "Expected 'sweep' (or a non-sample) dimension in sensitivity results."
                    )
                sweep_dim = cand[0]

            if x_override is not None:
                sweep = x_override
            else:
                sweep = (
                    np.asarray(line_data.coords[sweep_dim].values)
                    if sweep_dim in line_data.coords
                    else np.arange(line_data.sizes[sweep_dim])
                )

            mean = line_data.mean("sample") if "sample" in line_data.dims else line_data
            reduce_dims = [d for d in mean.dims if d != sweep_dim]
            if reduce_dims:
                mean = mean.sum(dim=reduce_dims)

            if "sample" in line_data.dims:
                hdi = az.hdi(line_data, hdi_prob=hdi_prob, input_core_dims=[["sample"]])
                if isinstance(hdi, xr.Dataset):
                    hdi = hdi[next(iter(hdi.data_vars))]
            else:
                hdi = xr.concat([mean, mean], dim="hdi").assign_coords(
                    hdi=np.array([0, 1])
                )

            reduce_hdi = [d for d in hdi.dims if d not in (sweep_dim, "hdi")]
            if reduce_hdi:
                hdi = hdi.sum(dim=reduce_hdi)
            if set(hdi.dims) == {sweep_dim, "hdi"} and list(hdi.dims) != [
                sweep_dim,
                "hdi",
            ]:
                hdi = hdi.transpose(sweep_dim, "hdi")  # type: ignore

            current_ax.plot(sweep, np.asarray(mean.values, dtype=float), **line_kwargs)
            az.plot_hdi(
                x=sweep,
                hdi_data=np.asarray(hdi.values, dtype=float),
                hdi_prob=hdi_prob,
                color=line_color,
                ax=current_ax,
            )

        # Get sweep coordinate values for absolute x-axis computation
        sweep_coord_values = None
        if x_sweep_axis == "absolute" and "sweep" in x.coords:
            sweep_coord_values = np.asarray(x.coords["sweep"].values)

        axes_flat = axes_array.flatten()
        for idx, combo in enumerate(dim_combinations):
            current_ax = axes_flat[idx]
            indexers = dict(zip(plot_dims, combo, strict=False)) if plot_dims else {}
            subset = x.sel(**indexers) if indexers else x

            if hue_dim is None:
                _plot_line(subset, _plot_kwargs, _line_color)
            else:
                for hue_value in hue_values:
                    hue_subset = subset.sel({hue_dim: hue_value})
                    line_kwargs = dict(_plot_kwargs)
                    if plot_kwargs is None or "color" not in plot_kwargs:
                        line_color = next(color_cycle)
                        line_kwargs["color"] = line_color
                    else:
                        line_color = line_kwargs.get("color", _line_color)
                    if "label" not in line_kwargs:
                        line_kwargs["label"] = f"{hue_dim}={hue_value}"

                    # Compute absolute x-values if requested
                    x_override = None
                    if x_sweep_axis == "absolute" and channel_scale is not None:
                        # Build indexers for channel_scale selection
                        scale_indexers = {**indexers, hue_dim: hue_value}
                        # Filter to only dims present in channel_scale
                        valid_scale_indexers = {
                            k: v
                            for k, v in scale_indexers.items()
                            if k in channel_scale.dims
                        }
                        try:
                            scale_value = float(
                                channel_scale.sel(**valid_scale_indexers).values
                            )
                            if sweep_coord_values is not None:
                                x_override = sweep_coord_values * scale_value
                        except (KeyError, ValueError) as err:
                            warnings.warn(
                                f"Could not compute absolute x-axis for "
                                f"{hue_dim}={hue_value}: {err}. "
                                f"Using relative sweep values.",
                                RuntimeWarning,
                                stacklevel=2,
                            )

                    _plot_line(hue_subset, line_kwargs, line_color, x_override)
                if legend_on:
                    current_ax.legend(**_legend_kwargs)

            title = self._build_subplot_title(
                dims=plot_dims,
                combo=combo,
                fallback_title=subplot_title_fallback,
            )
            current_ax.set_title(title)
            current_ax.set_xlabel(xlabel)
            current_ax.set_ylabel(ylabel)

        # Hide any unused axes (happens if grid > panels)
        for ax_extra in axes_flat[n_panels:]:
            ax_extra.set_visible(False)

        # Optional figure-level title: only for multi-panel layouts, default color (black)
        if add_figure_title and title is not None and n_panels > 1:
            fig.suptitle(title)

        if n_panels == 1:
            return axes_array[0, 0]

        fig.tight_layout()
        return fig, axes_array

    def uplift_curve(
        self,
        hdi_prob: float = 0.94,
        ax: plt.Axes | None = None,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        *,
        plot_kwargs: dict[str, Any] | None = None,
        ylabel: str = "Uplift",
        xlabel: str = "Sweep",
        title: str | None = "Uplift curve",
        add_figure_title: bool = True,
    ) -> tuple[Figure, NDArray[Axes]] | plt.Axes:
        """
        Plot precomputed uplift curves stored under `idata.sensitivity_analysis['uplift_curve']`.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        ax : plt.Axes, optional
            The axis to plot on.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        subplot_kwargs : dict, optional
            Additional subplot configuration forwarded to :meth:`sensitivity_analysis`.
        plot_kwargs : dict, optional
            Keyword arguments forwarded to the underlying line plot. If not provided, defaults
            are used by :meth:`sensitivity_analysis` (e.g., color "C0").
        ylabel : str, optional
            Y-axis label. Defaults to "Uplift".
        xlabel : str, optional
            X-axis label. Defaults to "Sweep".
        title : str, optional
            Figure-level title to add when ``add_figure_title=True``. Defaults to "Uplift curve".
        add_figure_title : bool, optional
            Whether to add a figure-level title. Defaults to ``True``.

        Examples
        --------
        Persist uplift curve and plot:

        .. code-block:: python

            from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

            sweeps = np.linspace(0.5, 1.5, 11)
            sa = SensitivityAnalysis(mmm.model, mmm.idata)
            results = sa.run_sweep(
                var_input="channel_data",
                sweep_values=sweeps,
                var_names="channel_contribution",
                sweep_type="multiplicative",
            )
            uplift = sa.compute_uplift_curve_respect_to_base(
                results, ref=1.0, extend_idata=True
            )
            _ = mmm.plot.uplift_curve(hdi_prob=0.9)
        """
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found in 'self.idata'. "
                "Run 'mmm.sensitivity.run_sweep()' first."
            )

        sa_group = self.idata.sensitivity_analysis  # type: ignore
        if isinstance(sa_group, xr.Dataset):
            if "uplift_curve" not in sa_group:
                raise ValueError(
                    "Expected 'uplift_curve' in idata.sensitivity_analysis. "
                    "Use SensitivityAnalysis.compute_uplift_curve_respect_to_base(..., extend_idata=True)."
                )
            data_var = sa_group["uplift_curve"]
        else:
            raise ValueError(
                "sensitivity_analysis does not contain 'uplift_curve'. Did you persist it to idata?"
            )

        # Delegate to a thin wrapper by temporarily constructing a Dataset
        tmp_idata = xr.Dataset({"x": data_var})
        # Monkey-patch minimal attributes needed
        tmp_idata["x"].attrs.update(getattr(sa_group, "attrs", {}))  # type: ignore
        # Temporarily swap
        original_group = self.idata.sensitivity_analysis  # type: ignore
        try:
            self.idata.sensitivity_analysis = tmp_idata  # type: ignore
            return self.sensitivity_analysis(
                hdi_prob=hdi_prob,
                ax=ax,
                aggregation=aggregation,
                subplot_kwargs=subplot_kwargs,
                subplot_title_fallback="Uplift curve",
                plot_kwargs=plot_kwargs,
                ylabel=ylabel,
                xlabel=xlabel,
                title=title,
                add_figure_title=add_figure_title,
            )
        finally:
            self.idata.sensitivity_analysis = original_group  # type: ignore

    def marginal_curve(
        self,
        hdi_prob: float = 0.94,
        ax: plt.Axes | None = None,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        *,
        plot_kwargs: dict[str, Any] | None = None,
        ylabel: str = "Marginal effect",
        xlabel: str = "Sweep",
        title: str | None = "Marginal effects",
        add_figure_title: bool = True,
    ) -> tuple[Figure, NDArray[Axes]] | plt.Axes:
        """
        Plot precomputed marginal effects stored under `idata.sensitivity_analysis['marginal_effects']`.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        ax : plt.Axes, optional
            The axis to plot on.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        subplot_kwargs : dict, optional
            Additional subplot configuration forwarded to :meth:`sensitivity_analysis`.
        plot_kwargs : dict, optional
            Keyword arguments forwarded to the underlying line plot. Defaults to ``{"color": "C1"}``.
        ylabel : str, optional
            Y-axis label. Defaults to "Marginal effect".
        xlabel : str, optional
            X-axis label. Defaults to "Sweep".
        title : str, optional
            Figure-level title to add when ``add_figure_title=True``. Defaults to "Marginal effects".
        add_figure_title : bool, optional
            Whether to add a figure-level title. Defaults to ``True``.

        Examples
        --------
        Persist marginal effects and plot:

        .. code-block:: python

            from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis

            sweeps = np.linspace(0.5, 1.5, 11)
            sa = SensitivityAnalysis(mmm.model, mmm.idata)
            results = sa.run_sweep(
                var_input="channel_data",
                sweep_values=sweeps,
                var_names="channel_contribution",
                sweep_type="multiplicative",
            )
            me = sa.compute_marginal_effects(results, extend_idata=True)
            _ = mmm.plot.marginal_curve(hdi_prob=0.9)
        """
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found in 'self.idata'. "
                "Run 'mmm.sensitivity.run_sweep()' first."
            )

        sa_group = self.idata.sensitivity_analysis  # type: ignore
        if isinstance(sa_group, xr.Dataset):
            if "marginal_effects" not in sa_group:
                raise ValueError(
                    "Expected 'marginal_effects' in idata.sensitivity_analysis. "
                    "Use SensitivityAnalysis.compute_marginal_effects(..., extend_idata=True)."
                )
            data_var = sa_group["marginal_effects"]
        else:
            raise ValueError(
                "sensitivity_analysis does not contain 'marginal_effects'. Did you persist it to idata?"
            )

        # We want a different y-label and color
        # Temporarily swap group to reuse plotting logic
        tmp = xr.Dataset({"x": data_var})
        tmp["x"].attrs.update(getattr(sa_group, "attrs", {}))  # type: ignore
        original = self.idata.sensitivity_analysis  # type: ignore
        try:
            self.idata.sensitivity_analysis = tmp  # type: ignore
            # Reuse core plotting; percentage=False by definition
            # Merge defaults for plot_kwargs if not provided
            _plot_kwargs = {"color": "C1"}
            if plot_kwargs:
                _plot_kwargs.update(plot_kwargs)
            return self.sensitivity_analysis(
                hdi_prob=hdi_prob,
                ax=ax,
                aggregation=aggregation,
                subplot_kwargs=subplot_kwargs,
                subplot_title_fallback="Marginal effects",
                plot_kwargs=_plot_kwargs,
                ylabel=ylabel,
                xlabel=xlabel,
                title=title,
                add_figure_title=add_figure_title,
            )
        finally:
            self.idata.sensitivity_analysis = original  # type: ignore

    def _process_decomposition_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data to compute the sum of contributions by component and calculate their percentages.

        The output dataframe will have columns for "component", "contribution", and "percentage".

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the contribution by component. Should have
            columns representing different components with numeric values.

        Returns
        -------
        pd.DataFrame
            A dataframe with contributions summed up by component, sorted by
            contribution in ascending order, with an additional column showing
            the percentage contribution of each component.
        """
        dataframe = data.copy()

        # Identify non-numeric columns to exclude (e.g., date and other dimension columns)
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in dataframe.columns if col not in numeric_cols]

        # Set non-numeric columns as index (if any) to exclude them from stacking
        if non_numeric_cols:
            dataframe = dataframe.set_index(non_numeric_cols)

        # Stack only the numeric contribution columns
        stack_dataframe = dataframe.stack().reset_index()

        # Determine column names based on number of index levels
        if len(non_numeric_cols) > 0:
            stack_dataframe.columns = pd.Index(
                [*non_numeric_cols, "component", "contribution"]
            )
            # Set index to include all non-numeric columns and component
            stack_dataframe.set_index([*non_numeric_cols, "component"], inplace=True)
        else:
            stack_dataframe.columns = pd.Index(["component", "contribution"])
            stack_dataframe.set_index(["component"], inplace=True)

        # Group by component and sum, which only affects the contribution column
        dataframe = stack_dataframe.groupby("component").sum(numeric_only=True)
        dataframe.sort_values(by="contribution", ascending=True, inplace=True)
        dataframe.reset_index(inplace=True)

        total_contribution = dataframe["contribution"].sum()
        dataframe["percentage"] = (dataframe["contribution"] / total_contribution) * 100

        return dataframe

    def _prepare_waterfall_data(
        self,
        var: list[str] | None = None,
        agg: str = "mean",
        dims: dict[str, str | int | list] | None = None,
        split_by: str | list[str] | None = None,
        original_scale: bool = True,
    ) -> tuple[pd.DataFrame, list[str], list[tuple]]:
        """Prepare data for waterfall plot with optional dimension filtering.

        This method handles data extraction, aggregation, and dimension processing
        for the waterfall components decomposition plot.

        Parameters
        ----------
        var : list of str, optional
            List of contribution variable names from the posterior to include.
            If None, automatically detects all contribution variables.
        agg : str, default "mean"
            Aggregation method for samples. Can be "mean" or "median".
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"}.
            If provided, only the selected slice(s) will be included.
        split_by : str or list of str, optional
            Dimension(s) to create separate subplots for. Each unique combination
            of values in these dimensions will get its own subplot.
        original_scale : bool, default True
            If True and var is None, use original scale contribution variables.
            If False and var is None, use non-original scale contribution variables.

        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame with contribution data, ready for processing.
        split_dims : list of str
            List of dimension names to split by.
        dim_combinations : list of tuple
            List of coordinate value combinations for split dimensions.

        Raises
        ------
        ValueError
            If no posterior data is found in idata.
            If none of the requested variables are present in idata.posterior.
            If split_by dimension is not found in the data.
        """
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure the model has been fitted."
            )

        # Auto-detect contribution variables if not specified
        if var is None:
            posterior_vars = list(self.idata.posterior.data_vars)
            # Variables to exclude - total_media_contribution is a sum of channels
            # and would double-count if included
            excluded_vars = {
                "total_media_contribution_original_scale",
                "total_media_contribution",
            }
            if original_scale:
                # Prefer original scale variables
                var = [
                    v
                    for v in posterior_vars
                    if v.endswith("_contribution_original_scale")
                    and v not in excluded_vars
                ]
                # If no original scale vars, fall back to regular contribution vars
                if not var:
                    var = [
                        v
                        for v in posterior_vars
                        if v.endswith("_contribution") and v not in excluded_vars
                    ]
            else:
                # Use non-original scale contribution variables
                var = [
                    v
                    for v in posterior_vars
                    if v.endswith("_contribution") and v not in excluded_vars
                ]

            if not var:
                raise ValueError(
                    "No contribution variables found in posterior. "
                    "Please specify the 'var' parameter explicitly."
                )

        # Build contributions DataFrame using the utility function
        dataframe = build_contributions(
            idata=self.idata,
            var=var,
            agg=agg,
        )

        # Apply dimension filtering if provided
        if dims:
            for key, val in dims.items():
                if key in dataframe.columns:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        dataframe = dataframe[dataframe[key].isin(val)]
                    else:
                        dataframe = dataframe[dataframe[key] == val]

        # Determine split dimensions and combinations
        if split_by is not None:
            split_dims = [split_by] if isinstance(split_by, str) else list(split_by)

            # Validate split dimensions exist in data
            for dim in split_dims:
                if dim not in dataframe.columns:
                    raise ValueError(
                        f"Split dimension '{dim}' not found in data columns. "
                        f"Available columns: {list(dataframe.columns)}"
                    )

            # Get unique combinations for split dimensions
            if split_dims:
                unique_values = [dataframe[dim].unique() for dim in split_dims]
                dim_combinations = list(itertools.product(*unique_values))
            else:
                dim_combinations = [()]
        else:
            split_dims = []
            dim_combinations = [()]

        return dataframe, split_dims, dim_combinations

    def _plot_single_waterfall(
        self,
        ax: Axes,
        data: pd.DataFrame,
        title: str = "Response Decomposition Waterfall by Components",
    ) -> None:
        """Plot a single waterfall chart on the given axes.

        This helper method renders a horizontal waterfall bar chart showing
        component contributions. It handles both positive and negative values,
        with cumulative positioning.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        data : pd.DataFrame
            DataFrame with columns "component", "contribution", and "percentage".
            Should be sorted by contribution in ascending order.
        title : str, default "Response Decomposition Waterfall by Components"
            Title for the subplot.
        """
        total_contribution = data["contribution"].sum()
        cumulative_contribution = 0

        for index, row in data.iterrows():
            color = "C0" if row["contribution"] >= 0 else "C3"

            bar_start = (
                cumulative_contribution + row["contribution"]
                if row["contribution"] < 0
                else cumulative_contribution
            )
            ax.barh(
                row["component"],
                row["contribution"],
                left=bar_start,
                color=color,
                alpha=0.5,
            )

            if row["contribution"] > 0:
                cumulative_contribution += row["contribution"]

            label_pos = bar_start + (row["contribution"] / 2)

            if row["contribution"] < 0:
                label_pos = bar_start - (row["contribution"] / 2)

            ax.text(
                label_pos,
                index,
                f"{row['contribution']:,.0f}\n({row['percentage']:.1f}%)",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

        ax.set_title(title)
        ax.set_xlabel("Cumulative Contribution")
        ax.set_ylabel("Components")

        if total_contribution > 0:
            xticks = np.linspace(0, total_contribution, num=11)
            xticklabels = [f"{(x / total_contribution) * 100:.0f}%" for x in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_yticks(np.arange(len(data)))
        ax.set_yticklabels(data["component"])

    def waterfall_components_decomposition(
        self,
        var: list[str] | None = None,
        original_scale: bool = True,
        dims: dict[str, str | int | list] | None = None,
        split_by: str | list[str] | None = None,
        figsize: tuple[int, int] | None = None,
        subplot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes] | tuple[Figure, NDArray[Axes]]:
        """Create a waterfall plot showing the decomposition of the target into its components.

        This plot visualizes how different model components (channels, controls, intercept,
        seasonality, etc.) contribute to the overall prediction. Each component is shown
        as a horizontal bar with its contribution value and percentage.

        Parameters
        ----------
        var : list of str, optional
            List of contribution variable names from the posterior to include in the plot.
            If None, automatically detects all contribution variables from the posterior.
            Example: ["intercept_contribution_original_scale",
                     "channel_contribution_original_scale",
                     "control_contribution_original_scale"]
        original_scale : bool, default True
            If True and var is None, use original scale contribution variables
            (ending with "_contribution_original_scale").
            If False and var is None, use non-original scale contribution variables.
            Ignored if var is explicitly provided.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"}.
            If provided, only the selected slice(s) will be included in the plot.
        split_by : str or list of str, optional
            Dimension(s) to create separate subplots for. Each unique combination
            of values in these dimensions will get its own waterfall plot.
            Example: "geo" or ["geo", "product"].
        figsize : tuple of int, optional
            The size of the figure in inches (width, height).
            If None, defaults to (14, 7) for single plots, or auto-calculated
            based on number of subplots.
        subplot_kwargs : dict, optional
            Additional keyword arguments for subplot layout configuration.
            Supports "nrows" or "ncols" to control grid arrangement.
            Only one of nrows or ncols should be specified.
        **kwargs
            Additional keyword arguments passed to matplotlib's `subplots` function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot(s).
        ax_or_axes : matplotlib.axes.Axes or np.ndarray of Axes
            If split_by is None: single Axes object.
            If split_by is provided: 2D array of Axes objects.

        Raises
        ------
        ValueError
            If no posterior data is found in idata.
            If none of the requested variables are present in idata.posterior.
            If split_by dimension is not found in the data.
            If both nrows and ncols are specified in subplot_kwargs.

        Examples
        --------
        Create a waterfall plot with all contribution variables (auto-detected):

        .. code-block:: python

            fig, ax = mmm.plot.waterfall_components_decomposition()

        Create a waterfall plot with specific contribution variables:

        .. code-block:: python

            fig, ax = mmm.plot.waterfall_components_decomposition(
                var=[
                    "intercept_contribution_original_scale",
                    "channel_contribution_original_scale",
                    "control_contribution_original_scale",
                ]
            )

        With custom figure size:

        .. code-block:: python

            fig, ax = mmm.plot.waterfall_components_decomposition(figsize=(18, 10))

        Filter by dimension:

        .. code-block:: python

            fig, ax = mmm.plot.waterfall_components_decomposition(dims={"geo": "US"})

        Create subplots split by dimension:

        .. code-block:: python

            fig, axes = mmm.plot.waterfall_components_decomposition(split_by="geo")

        Control subplot layout:

        .. code-block:: python

            fig, axes = mmm.plot.waterfall_components_decomposition(
                split_by="geo",
                subplot_kwargs={"ncols": 2},
            )
        """
        # Prepare the data with filtering and dimension handling
        dataframe, split_dims, dim_combinations = self._prepare_waterfall_data(
            var=var,
            agg="mean",
            dims=dims,
            split_by=split_by,
            original_scale=original_scale,
        )

        n_panels = len(dim_combinations)
        subplot_kwargs = {**(subplot_kwargs or {})}

        # Handle subplot grid configuration
        nrows_user = subplot_kwargs.pop("nrows", None)
        ncols_user = subplot_kwargs.pop("ncols", None)

        if nrows_user is not None and ncols_user is not None:
            raise ValueError(
                "Specify only one of 'nrows' or 'ncols' in subplot_kwargs."
            )

        # Single panel case (no split_by or single combination)
        if n_panels == 1:
            if figsize is None:
                figsize = (14, 7)

            fig, ax = plt.subplots(
                figsize=figsize, layout="constrained", **subplot_kwargs, **kwargs
            )

            # Process data and plot
            processed_data = self._process_decomposition_components(data=dataframe)
            self._plot_single_waterfall(
                ax=ax,
                data=processed_data,
                title="Response Decomposition Waterfall by Components",
            )

            return fig, ax

        # Multiple panels case (split_by provided)
        if ncols_user is not None:
            ncols = ncols_user
            nrows = int(np.ceil(n_panels / ncols))
        elif nrows_user is not None:
            nrows = nrows_user
            ncols = int(np.ceil(n_panels / nrows))
        else:
            ncols = max(1, int(np.ceil(np.sqrt(n_panels))))
            nrows = int(np.ceil(n_panels / ncols))

        if figsize is None:
            figsize = (ncols * 10, nrows * 6)

        fig, axes_grid = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            layout="constrained",
            **subplot_kwargs,
            **kwargs,
        )

        # Normalize axes to 2D array
        if isinstance(axes_grid, Axes):
            axes_array = np.array([[axes_grid]])
        elif axes_grid.ndim == 1:
            axes_array = (
                axes_grid.reshape(1, -1) if nrows == 1 else axes_grid.reshape(-1, 1)
            )
        else:
            axes_array = axes_grid

        # Flatten for iteration
        axes_flat = axes_array.flatten()

        # Plot each combination
        for idx, combo in enumerate(dim_combinations):
            current_ax = axes_flat[idx]

            # Filter data for this combination
            if split_dims:
                mask = pd.Series(True, index=dataframe.index)
                for dim, val in zip(split_dims, combo, strict=False):
                    mask &= dataframe[dim] == val
                subset = dataframe[mask].copy()
            else:
                subset = dataframe.copy()

            # Process and plot
            processed_data = self._process_decomposition_components(data=subset)

            # Build subplot title
            title = self._build_subplot_title(
                dims=split_dims,
                combo=combo,
                fallback_title="Response Decomposition Waterfall by Components",
            )

            self._plot_single_waterfall(
                ax=current_ax,
                data=processed_data,
                title=title,
            )

        # Hide unused axes
        for ax_extra in axes_flat[n_panels:]:
            ax_extra.set_visible(False)

        return fig, axes_array

    def channel_contribution_share_hdi(
        self,
        hdi_prob: float = 0.94,
        dims: dict[str, str | int | list] | None = None,
        figsize: tuple[float, float] = (10, 6),
        **plot_kwargs: Any,
    ) -> tuple[Figure, Axes]:
        """Plot the share of channel contributions in a forest plot.

        Shows the percentage contribution of each channel to the total response,
        computed from channel contributions in the original scale. Each channel's
        share represents what percentage of the total response it accounts for.

        Parameters
        ----------
        hdi_prob : float, optional
            HDI probability mass to display. Default is 0.94.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"geo": "US"}.
            If provided, only the selected slice(s) will be plotted.
        figsize : tuple[float, float], optional
            Figure size. Default is (10, 6).
        **plot_kwargs
            Additional keyword arguments passed to `az.plot_forest`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object with the forest plot.

        Raises
        ------
        ValueError
            If `channel_contribution_original_scale` is not found in posterior.
            If no posterior data is found in idata.

        Examples
        --------
        Plot channel contribution shares:

        .. code-block:: python

            fig, ax = mmm.plot.channel_contribution_share_hdi(hdi_prob=0.94)

        With dimension filtering:

        .. code-block:: python

            fig, ax = mmm.plot.channel_contribution_share_hdi(
                hdi_prob=0.90, dims={"geo": "US"}
            )
        """
        # Check if posterior exists
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure the model has been fitted."
            )

        # Check if channel_contribution_original_scale exists
        if "channel_contribution_original_scale" not in self.idata.posterior:
            raise ValueError(
                "Variable 'channel_contribution_original_scale' not found in posterior. "
                "Add it using:\n"
                "    mmm.add_original_scale_contribution_variable(\n"
                "        var=['channel_contribution']\n"
                "    )"
            )

        # Extract the variable
        channel_contribution_original_scale = az.extract(
            data=self.idata.posterior,
            var_names=["channel_contribution_original_scale"],
            combined=False,
        )

        # Apply dimension filtering if provided
        if dims:
            all_dims = list(channel_contribution_original_scale.dims)
            self._validate_dims(dims=dims, all_dims=all_dims)

            # Build indexers for filtering
            indexers = {}
            for key, val in dims.items():
                if key in all_dims:
                    indexers[key] = val

            if indexers:
                channel_contribution_original_scale = (
                    channel_contribution_original_scale.sel(**indexers)
                )

        # Sum over date dimension to get total per channel
        if "date" in channel_contribution_original_scale.dims:
            numerator = channel_contribution_original_scale.sum(["date"])
        else:
            numerator = channel_contribution_original_scale

        # Divide by sum across channels to get share
        if "channel" in numerator.dims:
            denominator = numerator.sum("channel")
            channel_contribution_share = numerator / denominator
        else:
            raise ValueError(
                "Expected 'channel' dimension in channel_contribution_original_scale, "
                "but none found."
            )

        # Create the forest plot
        ax, *_ = az.plot_forest(
            data=channel_contribution_share,
            combined=True,
            hdi_prob=hdi_prob,
            figsize=figsize,
            **plot_kwargs,
        )

        # Format x-axis as percentages
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y: 0.0%}"))

        # Get the figure and set title
        fig: Figure = plt.gcf()
        fig.suptitle("Channel Contribution Share", fontsize=16, y=1.05)

        return fig, ax

    def cv_predictions(
        self, results: az.InferenceData, dims: dict[str, str | int | list] | None = None
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot posterior predictive predictions across CV folds.

        Generates visualization showing posterior predictive distributions for
        each cross-validation fold, with separate panels for different dimension
        combinations.

        Parameters
        ----------
        results : arviz.InferenceData
            Combined InferenceData produced by ``TimeSliceCrossValidator.run()``.
            Must contain:

            - A coordinate named 'cv'
            - A 'cv_metadata' group with per-fold metadata (X_train, y_train,
              X_test, y_test) stored under ``cv_metadata.metadata``
            - A posterior_predictive group containing 'y_original_scale'

        dims : dict, optional
            Dictionary specifying dimensions to filter when plotting.
            Keys must be coordinates present on
            ``posterior_predictive['y_original_scale']``.
            Values can be single values or lists of values.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        axes : numpy.ndarray of matplotlib.axes.Axes
            Array of axes objects, one per panel.

        Raises
        ------
        TypeError
            If ``results`` is not an ``arviz.InferenceData`` object.
        ValueError
            If required groups or variables are missing from ``results``.
            If unsupported dimensions are specified in ``dims``.

        Notes
        -----
        The plot shows:

        - HDI (94%) for train (blue) and test (orange) ranges as shaded bands
        - Observed values as black lines
        - A vertical dashed green line marking the end of training for each fold

        See Also
        --------
        TimeSliceCrossValidator.run : Generate the combined InferenceData.
        param_stability : Plot parameter stability across folds.
        cv_crps : Plot CRPS scores across folds.
        """
        # Expect an arviz.InferenceData with cv coord and cv_metadata
        if not isinstance(results, az.InferenceData):
            raise TypeError(
                "plot_cv_predictions expects an arviz.InferenceData object for 'results'."
            )

        # Validate presence of cv metadata and posterior predictive
        if not hasattr(results, "cv_metadata") or "metadata" not in results.cv_metadata:
            raise ValueError(
                "Provided InferenceData must include a 'cv_metadata' group with a 'metadata' DataArray."
            )
        if (
            not hasattr(results, "posterior_predictive")
            or "y_original_scale" not in results.posterior_predictive
        ):
            raise ValueError(
                "Provided InferenceData must include posterior_predictive['y_original_scale']."
            )

        # Discover posterior_predictive dataarray we'll be working with
        pp = results.posterior_predictive["y_original_scale"]

        # Determine which coordinate dims are available for paneling (exclude technical dims)
        technical_dims = {"chain", "draw", "sample", "date", "cv"}
        available_dims = [d for d in pp.dims if d not in technical_dims]

        # If the user supplied dims, validate they are a subset of available_dims
        if dims is None:
            # Require explicit dims or use additional dims for paneling
            dims = {}
        else:
            unsupported = [d for d in dims.keys() if d not in available_dims]
            if unsupported:
                raise ValueError(
                    f"cv_predictions only supports dims that exist. Unsupported dims: {unsupported}"
                )

        # Build explicit lists for dims that may contain single values
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # Additional dimensions to create separate panels for (those not in dims and not ignored)
        additional_dims = [d for d in available_dims if d not in dims_keys]
        if additional_dims:
            additional_coords = [pp.coords[d].values for d in additional_dims]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        # Build all panel indexers: each panel corresponds to a mapping dim->value
        total_panels = []
        for dims_combo in dims_combos:
            for addl_combo in additional_combinations:
                indexer: dict = {}
                indexer.update(dict(zip(dims_keys, dims_combo, strict=False)))
                if additional_dims:
                    indexer.update(dict(zip(additional_dims, addl_combo, strict=False)))
                total_panels.append(indexer)

        cv_labels = list(results.cv_metadata.coords["cv"].values)
        n_folds = len(cv_labels)
        n_panels = len(total_panels)
        n_axes = max(1, n_panels * n_folds)

        fig, axes = plt.subplots(
            n_axes, 1, figsize=(12, 4 * max(1, n_axes)), sharex=True
        )
        if n_axes == 1:
            axes = [axes]

        # Helper to align y Series to a DataFrame's rows without using reindex (avoids duplicate-index errors)
        def _align_y_to_df(y_series, df):
            y_df = y_series.reset_index()
            y_df.columns = ["orig_index", "y_value"]
            df_idx = pd.DataFrame({"orig_index": df.index, "date": df["date"].values})
            merged = df_idx.merge(y_df, on="orig_index", how="left")
            return merged["y_value"], merged["date"]

        # Robust wrapper to call arviz.plot_hdi from an xarray DataArray `sel`.
        def _plot_hdi_from_sel(sel, ax, color, label):
            sel2 = sel.squeeze()
            arr = getattr(sel2, "values", sel2)
            if arr.ndim == 1:
                if hasattr(sel2, "coords") and "sample" in sel2.coords:
                    arr = arr.reshape((-1, 1))
                    x = (
                        sel2.coords["date"].values
                        if "date" in sel2.coords
                        else [sel2.coords.get("date")]
                    )
                else:
                    arr = arr.reshape((1, -1))
                    x = (
                        sel2.coords["date"].values
                        if "date" in sel2.coords
                        else [sel2.coords.get("date")]
                    )
            else:
                if hasattr(sel2, "dims"):
                    dims = list(sel2.dims)
                    if dims == ["date", "sample"]:
                        arr = arr.T
                    elif dims != ["sample", "date"]:
                        try:
                            sel2 = sel2.transpose("sample", "date")
                            arr = sel2.values
                        except Exception as exc:
                            warnings.warn(
                                f"Could not transpose sel2 to ('sample','date'): {exc}",
                                stacklevel=2,
                            )
                            arr = getattr(sel2, "values", sel2)
                x = (
                    sel2.coords["date"].values
                    if hasattr(sel2, "coords") and "date" in sel2.coords
                    else None
                )

            # Ensure x is at least 1D array (arviz.plot_hdi fails on 0-dim arrays)
            if x is not None:
                x = np.atleast_1d(x)

            az.plot_hdi(
                y=arr,
                x=x,
                ax=ax,
                hdi_prob=0.94,
                color=color,
                smooth=False,
                fill_kwargs={"alpha": 0.25, "label": label},
                plot_kwargs={"color": color, "linestyle": "--", "linewidth": 1},
            )

        # Iterate panels x folds
        for panel_idx, panel_indexer in enumerate(total_panels):
            for fold_idx, cv_label in enumerate(cv_labels):
                ax_i = panel_idx * n_folds + fold_idx
                ax = axes[ax_i]

                # Select posterior predictive array for this CV and this panel
                arr = results.posterior_predictive["y_original_scale"].sel(cv=cv_label)
                try:
                    arr = arr.sel(**panel_indexer) if panel_indexer else arr
                except (KeyError, ValueError) as exc:
                    # If a specific panel coord value cannot be selected (e.g., not present), warn and skip
                    warnings.warn(
                        f"Could not select posterior_predictive panel {panel_indexer}: {exc}; skipping.",
                        stacklevel=2,
                    )
                    continue

                # Stack chain/draw -> sample for quantile computation and ensure ordering
                arr_s = arr.stack(sample=("chain", "draw"))
                # Ensure date is a dimension we can index into and keep ordering date,last dims
                # Move date and sample to front for consistent indexing used by helper
                try:
                    arr_s = arr_s.transpose(
                        "sample",
                        "date",
                        *[d for d in arr_s.dims if d not in ("sample", "date")],
                    )
                except (ValueError, KeyError) as exc:
                    # If transpose fails, continue with whatever ordering exists
                    warnings.warn(
                        f"Could not transpose posterior_predictive array to ('sample','date',...): {exc}",
                        stacklevel=2,
                    )

                # Extract train/test metadata for this fold from cv_metadata
                meta_da = results.cv_metadata["metadata"].sel(cv=cv_label)
                try:
                    meta = meta_da.values.item()
                except (ValueError, AttributeError):
                    # fallback: try python object access
                    meta = getattr(meta_da, "item", lambda: None)()

                X_train = meta.get("X_train") if isinstance(meta, dict) else None
                y_train = meta.get("y_train") if isinstance(meta, dict) else None
                X_test = meta.get("X_test") if isinstance(meta, dict) else None
                y_test = meta.get("y_test") if isinstance(meta, dict) else None

                # Filter train/test DataFrames to this panel
                train_df_panel = self._filter_df_by_indexer(X_train, panel_indexer)
                test_df_panel = self._filter_df_by_indexer(X_test, panel_indexer)

                train_dates = (
                    pd.to_datetime(train_df_panel["date"].values)
                    if not train_df_panel.empty
                    else pd.DatetimeIndex([])
                )
                test_dates = (
                    pd.to_datetime(test_df_panel["date"].values)
                    if not test_df_panel.empty
                    else pd.DatetimeIndex([])
                )
                train_dates = train_dates.sort_values().unique()
                test_dates = test_dates.sort_values().unique()

                # Plot HDI for train (blue) and test (orange)
                if train_dates.size:
                    try:
                        sel = arr_s.sel(
                            date=train_dates,
                            **{
                                k: v
                                for k, v in panel_indexer.items()
                                if k in arr_s.dims
                            },
                        )
                        _plot_hdi_from_sel(sel, ax, "C0", "HDI (train)")
                    except (KeyError, ValueError, TypeError) as exc:
                        warnings.warn(
                            f"Could not compute HDI for train range: {exc}; skipping.",
                            stacklevel=2,
                        )

                if test_dates.size:
                    try:
                        sel = arr_s.sel(
                            date=test_dates,
                            **{
                                k: v
                                for k, v in panel_indexer.items()
                                if k in arr_s.dims
                            },
                        )
                        _plot_hdi_from_sel(sel, ax, "C1", "HDI (test)")
                    except (KeyError, ValueError, TypeError) as exc:
                        warnings.warn(
                            f"Could not compute HDI for test range: {exc}; skipping.",
                            stacklevel=2,
                        )

                # Plot observed actuals in black (train + test) as lines (no markers)
                if (
                    X_train is not None
                    and y_train is not None
                    and not train_df_panel.empty
                ):
                    y_train_vals, train_plot_dates = _align_y_to_df(
                        y_train, train_df_panel
                    )
                    y_train_vals = y_train_vals.dropna()
                    if not y_train_vals.empty:
                        dates_to_plot = pd.to_datetime(
                            train_plot_dates.loc[y_train_vals.index].values
                        )
                        ax.plot(
                            dates_to_plot,
                            y_train_vals.values,
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                            label="observed",
                        )

                if (
                    X_test is not None
                    and y_test is not None
                    and not test_df_panel.empty
                ):
                    y_test_vals, test_plot_dates = _align_y_to_df(y_test, test_df_panel)
                    y_test_vals = y_test_vals.dropna()
                    if not y_test_vals.empty:
                        dates_to_plot = pd.to_datetime(
                            test_plot_dates.loc[y_test_vals.index].values
                        )
                        ax.plot(
                            dates_to_plot,
                            y_test_vals.values,
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                        )

                # Vertical line marking end of training
                if train_dates.size:
                    end_train_date = pd.to_datetime(train_dates.max())
                    ax.axvline(
                        end_train_date,
                        color="green",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.9,
                        label="train end",
                    )

                # Build title from panel indexer values
                if panel_indexer:
                    title_parts = [f"{k}={v}" for k, v in panel_indexer.items()]
                    panel_title = ", ".join(title_parts)
                else:
                    panel_title = "Posterior Predictive"
                ax.set_title(f"{panel_title} — Fold {fold_idx} — Posterior Predictive")
                ax.set_ylabel("y_original_scale")

        # Build a single unique legend placed at the bottom of the figure
        handles, labels = [], []
        for ax in axes:
            h, _l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(_l)
        by_label = dict(zip(labels, handles, strict=False))
        if by_label:
            plt.tight_layout(rect=[0, 0.07, 1, 1])
            ncol = min(4, len(by_label))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.01),
            )
        else:
            plt.tight_layout()

        axes[-1].set_xlabel("date")
        plt.show()
        return fig, axes

    def param_stability(
        self,
        results: az.InferenceData,
        parameter: list[str],
        dims: dict[str, list[str]] | None = None,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot parameter stability across CV iterations.

        Generates forest plots showing how parameter estimates vary across
        cross-validation folds, helping assess model stability.

        Parameters
        ----------
        results : arviz.InferenceData
            Combined InferenceData produced by ``TimeSliceCrossValidator.run()``.
            Must contain a coordinate named 'cv' which labels each CV fold.
        parameter : list of str
            List of parameter names to plot (e.g., ``["beta_channel"]``).
        dims : dict, optional
            Dictionary specifying dimensions and coordinate values to slice over.
            Each key is a dimension name, and the value is a list of coordinate
            values. A separate forest plot is generated for each combination.
            Example: ``{"geo": ["geo_a", "geo_b"]}``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object (last one if multiple plots generated).
        ax : matplotlib.axes.Axes
            The axes object (last one if multiple plots generated).

        Raises
        ------
        TypeError
            If ``results`` is not an ``arviz.InferenceData`` object.
        ValueError
            If the InferenceData does not contain a 'cv' coordinate.
            If unable to select specified dimensions from posterior.

        See Also
        --------
        TimeSliceCrossValidator.run : Generate the combined InferenceData.
        cv_predictions : Plot posterior predictive across folds.
        cv_crps : Plot CRPS scores across folds.

        Examples
        --------
        Basic usage:

        >>> suite = MMMPlotSuite(idata=None)
        >>> fig, ax = suite.param_stability(combined_idata, parameter=["beta_channel"])

        With dimension slicing:

        >>> fig, ax = suite.param_stability(
        ...     combined_idata, parameter=["beta_channel"], dims={"geo": ["US", "UK"]}
        ... )
        """
        # Ensure the provided input is an arviz.InferenceData with a 'cv' coord
        if not isinstance(results, az.InferenceData):
            raise TypeError(
                "plot_param_stability expects an `arviz.InferenceData` returned by TimeSliceCrossValidator.run(...)"
            )

        idata = results
        # discover cv labels from any group that exposes the coordinate
        cv_labels = None
        for grp in (
            "posterior",
            "posterior_predictive",
            "sample_stats",
            "observed_data",
            "prior",
        ):
            try:
                ds = getattr(idata, grp)
            except AttributeError:
                ds = None
            if ds is None:
                continue
            if "cv" in ds.coords:
                cv_labels = list(ds.coords["cv"].values)
                break

        if cv_labels is None:
            raise ValueError(
                "Provided InferenceData does not contain a 'cv' coordinate."
            )

        # Build posterior_list by selecting along cv for the posterior group
        posterior_list = []
        model_names: list[str] = []
        for lbl in cv_labels:
            try:
                p = idata.posterior.sel(cv=lbl)
            except (KeyError, AttributeError):
                # fallback to selecting from posterior_predictive if posterior missing
                p = idata.posterior_predictive.sel(cv=lbl)

            posterior_list.append(p)
            model_names.append(str(lbl))

        if dims is None:
            # No dims: standard forest plot
            fig, ax = plt.subplots(figsize=(9, 6))
            az.plot_forest(
                data=posterior_list,
                model_names=model_names,
                var_names=parameter,
                combined=True,
                ax=ax,
            )
            fig.suptitle(
                f"Parameter Stability: {parameter}",
                fontsize=18,
                fontweight="bold",
                y=1.06,
            )
            plt.show()
            return fig, ax

        else:
            # Plot one forest plot per dim value
            last_fig_ax = None
            for dim_name, coord_values in dims.items():
                for coord in coord_values:
                    fig, ax = plt.subplots(figsize=(9, 6))
                    # Select the coordinate value from each posterior fold
                    sel_data = []
                    for p in posterior_list:
                        try:
                            sel_data.append(p.sel({dim_name: coord}))
                        except (KeyError, ValueError) as exc:
                            raise ValueError(
                                f"Unable to select dims from posterior for one or more folds: {exc}"  # noqa: S608
                            ) from exc

                    az.plot_forest(
                        data=sel_data,
                        model_names=model_names,
                        var_names=parameter,
                        combined=True,
                        ax=ax,
                    )
                    fig.suptitle(
                        f"Parameter Stability: {parameter} | {dim_name}={coord}",
                        fontsize=18,
                        fontweight="bold",
                        y=1.06,
                    )
                    plt.show()
                    last_fig_ax = (fig, ax)

            # If dims provided but empty, fall back to the no-dims behavior
            if last_fig_ax is None:
                fig, ax = plt.subplots(figsize=(9, 6))
                az.plot_forest(
                    data=posterior_list,
                    model_names=model_names,
                    var_names=parameter,
                    combined=True,
                    ax=ax,
                )
                fig.suptitle(
                    f"Parameter Stability: {parameter}",
                    fontsize=18,
                    fontweight="bold",
                    y=1.06,
                )
                plt.show()
                return fig, ax

            return last_fig_ax

    def cv_crps(
        self, results: az.InferenceData, dims: dict[str, str | int | list] | None = None
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot CRPS scores for train and test sets across CV splits.

        Generates plots showing the Continuous Ranked Probability Score (CRPS)
        for each cross-validation fold, optionally stratified by additional
        dimensions.

        Parameters
        ----------
        results : arviz.InferenceData
            Combined InferenceData produced by ``TimeSliceCrossValidator.run()``.
            Must contain:

            - A coordinate named 'cv'
            - A 'cv_metadata' group with per-fold metadata (X_train, y_train,
              X_test, y_test) stored under ``cv_metadata.metadata``
            - A posterior_predictive group containing 'y_original_scale'

        dims : dict, optional
            Dictionary specifying dimensions to stratify the CRPS computation.
            Keys must be coordinates present on
            ``posterior_predictive['y_original_scale']``.
            Values can be single values or lists of values.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        axes : numpy.ndarray of matplotlib.axes.Axes
            2D array of axes objects with shape (n_panels, 2), where the first
            column shows train CRPS and the second shows test CRPS.

        Raises
        ------
        TypeError
            If ``results`` is not an ``arviz.InferenceData`` object.
        ValueError
            If required groups or variables are missing from ``results``.
            If no 'cv' coordinate is found in the InferenceData.

        See Also
        --------
        TimeSliceCrossValidator.run : Generate the combined InferenceData.
        cv_predictions : Plot posterior predictive across folds.
        param_stability : Plot parameter stability across folds.

        Notes
        -----
        CRPS (Continuous Ranked Probability Score) is a proper scoring rule
        that measures the quality of probabilistic predictions. Lower values
        indicate better predictions.
        """
        # Validate input is combined InferenceData
        if not isinstance(results, az.InferenceData):
            raise TypeError(
                "cv_crps expects an arviz.InferenceData returned by TimeSliceCrossValidator._combine_idata(...)"
            )
        if not hasattr(results, "cv_metadata") or "metadata" not in results.cv_metadata:
            raise ValueError(
                "Provided InferenceData must include a 'cv_metadata' group with a 'metadata' DataArray."
            )
        if (
            not hasattr(results, "posterior_predictive")
            or "y_original_scale" not in results.posterior_predictive
        ):
            raise ValueError(
                "Provided InferenceData must include posterior_predictive['y_original_scale']."
            )

        # Helper: build prediction matrix for a given cv label and rows DataFrame
        def _pred_matrix_for_rows(
            idata: az.InferenceData, cv_label, rows_df: pd.DataFrame
        ):
            """Build (n_samples, n_rows) prediction matrix for given rows DataFrame and CV label.

            Selects posterior_predictive['y_original_scale'] for the given cv and then
            behaves like the legacy helper: find date coord, select by date (and any
            other matching row-level coords), and assemble a (n_samples, n_rows) matrix.
            """
            da = idata.posterior_predictive["y_original_scale"].sel(cv=cv_label)
            da_s = da.stack(sample=("chain", "draw"))

            # Ensure 'sample' is first axis
            if da_s.dims[0] != "sample":
                da_s = da_s.transpose("sample", ...)
            else:
                dims = list(da_s.dims)
                order = ["sample"] + [d for d in dims if d != "sample"]
                da_s = da_s.transpose(*order)

            n_samples = int(da_s.sizes["sample"])
            n_rows = len(rows_df)
            mat = np.empty((n_samples, n_rows))

            for j, (_idx, row) in enumerate(rows_df.iterrows()):
                # determine date coord
                date_coord = None
                if "date" in da_s.coords:
                    date_coord = "date"
                else:
                    technical_skip = {"sample", "chain", "draw"}
                    for coord, vals in da_s.coords.items():
                        if coord in technical_skip:
                            continue
                        if pd.api.types.is_datetime64_any_dtype(
                            getattr(vals, "dtype", vals)
                        ):
                            date_coord = coord
                            break

                if date_coord is None:
                    for coord in da_s.coords:
                        if coord not in ("sample", "chain", "draw"):
                            date_coord = coord
                            break

                # find matching row date value
                if date_coord in rows_df.columns:
                    date_value = row[date_coord]  # type: ignore[index]
                else:
                    found_col = None
                    for col in rows_df.columns:
                        if "date" in col.lower():
                            found_col = col
                            break
                    if found_col is None:
                        for col in rows_df.columns:
                            if pd.api.types.is_datetime64_any_dtype(rows_df[col].dtype):
                                found_col = col

                    if found_col is None:
                        raise ValueError(
                            "Could not find a date-like column in rows_df to match posterior_predictive coordinate"
                        )
                    # found_col is guaranteed to be str here after the check above
                    date_value = row[found_col]  # type: ignore[index]

                # select by date
                sel = da_s.sel({date_coord: date_value})

                # select by any other dims that appear in both sel.dims and rows_df.columns
                other_dims = [d for d in sel.dims if d not in ("sample", date_coord)]
                for dim in other_dims:
                    if dim in rows_df.columns:
                        try:
                            sel = sel.sel({dim: str(row[dim])})
                        except (KeyError, ValueError):
                            # try without casting to string if that fails
                            sel = sel.sel({dim: row[dim]})

                arr = np.squeeze(getattr(sel, "values", sel))
                if arr.ndim == 0:
                    raise ValueError(
                        "Posterior predictive selection returned a scalar for a row"
                    )
                if arr.ndim > 1:
                    arr = arr.reshape(n_samples, -1)[:, 0]

                mat[:, j] = arr

            return mat

        # dims handling (validate + build combinations)
        # derive dims from the posterior_predictive (use first cv to inspect dims)
        # discover cv labels from cv_metadata (preferred) or posterior_predictive coords
        if hasattr(results, "cv_metadata") and "cv" in results.cv_metadata.coords:
            cv_labels = list(results.cv_metadata.coords["cv"].values)
        elif (
            hasattr(results, "posterior_predictive")
            and "cv" in results.posterior_predictive.coords
        ):
            cv_labels = list(results.posterior_predictive.coords["cv"].values)
        else:
            raise ValueError(
                "No 'cv' coordinate found in provided InferenceData (checked cv_metadata and posterior_predictive)"
            )
        if not cv_labels:
            raise ValueError("No CV labels found in provided InferenceData")
        main_da = results.posterior_predictive["y_original_scale"].sel(cv=cv_labels[0])
        all_dims = list(main_da.dims)

        # validate dims
        if dims:
            self._validate_dims(dims, all_dims)
        else:
            self._validate_dims({}, all_dims)
        dims_keys, dims_combos = self._dim_list_handler(dims)

        # identify additional dims to iterate over
        ignored_dims = {"chain", "draw", "sample", "date"}
        additional_dims = [
            d for d in all_dims if d not in ignored_dims and d not in (dims or {})
        ]
        if additional_dims:
            additional_coords = [main_da.coords[d].values for d in additional_dims]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        total_combos = list(itertools.product(dims_combos, additional_combinations))
        n_panels = len(total_combos)

        # create one panel per combination -> create two columns: train | test
        fig, axes = self._init_subplots(n_subplots=max(1, n_panels), ncols=2)

        def _filter_rows_and_y(
            df: pd.DataFrame | None, y: pd.Series | None, indexers: dict
        ) -> tuple[pd.DataFrame, np.ndarray]:
            # Accept optional df and y to satisfy callers; always return concrete types
            if df is None or df.empty:
                return pd.DataFrame([], columns=[]), np.array([])
            if y is None:
                return pd.DataFrame([], columns=[]), np.array([])
            mask = np.ones(len(df), dtype=bool)
            for k, v in indexers.items():
                if k in df.columns:
                    mask &= df[k].astype(str) == str(v)
            filtered_df = df[mask].reset_index(drop=True)
            y_arr = y.to_numpy()[mask]

            return filtered_df, y_arr

        # iterate and compute per-panel CRPS across folds
        for panel_idx, (dims_combo, addl_combo) in enumerate(total_combos):
            ax_train = axes[panel_idx][0]
            ax_test = axes[panel_idx][1]
            indexers: dict = {}
            for i, k in enumerate(dims_keys):
                indexers[k] = dims_combo[i]
            for k, v in (dims or {}).items():
                if k not in dims_keys:
                    indexers[k] = v
            for i, d in enumerate(additional_dims):
                indexers[d] = addl_combo[i] if addl_combo else addl_combo

            crps_train_list = []
            crps_test_list = []

            # loop over cv folds using cv_metadata
            for _cv_idx, cv_label in enumerate(cv_labels):
                meta_da = results.cv_metadata["metadata"].sel(cv=cv_label)
                vals = getattr(meta_da, "values", None)
                if isinstance(vals, np.ndarray) and vals.size == 1:
                    meta = vals.item()
                else:
                    meta = getattr(meta_da, "item", lambda: None)()

                X_train_df = meta.get("X_train") if isinstance(meta, dict) else None
                y_train = meta.get("y_train") if isinstance(meta, dict) else None
                X_test_df = meta.get("X_test") if isinstance(meta, dict) else None
                y_test = meta.get("y_test") if isinstance(meta, dict) else None

                # Training data
                filtered_train_rows, y_train_arr = _filter_rows_and_y(
                    X_train_df, y_train, indexers
                )
                if len(filtered_train_rows) == 0:
                    crps_train_list.append(np.nan)
                else:
                    try:
                        y_pred_train = _pred_matrix_for_rows(
                            results,
                            cv_label,
                            filtered_train_rows.reset_index(drop=True),
                        )
                        if y_pred_train.shape[1] != len(y_train_arr):
                            crps_train_list.append(np.nan)
                        else:
                            crps_train_list.append(
                                crps(y_true=y_train_arr, y_pred=y_pred_train)
                            )
                    except (KeyError, ValueError, IndexError):
                        crps_train_list.append(np.nan)

                # Testing data
                filtered_test_rows, y_test_arr = _filter_rows_and_y(
                    X_test_df, y_test, indexers
                )
                if len(filtered_test_rows) == 0:
                    crps_test_list.append(np.nan)
                else:
                    try:
                        y_pred_test = _pred_matrix_for_rows(
                            results, cv_label, filtered_test_rows.reset_index(drop=True)
                        )
                        if y_pred_test.shape[1] != len(y_test_arr):
                            crps_test_list.append(np.nan)
                        else:
                            crps_test_list.append(
                                crps(y_true=y_test_arr, y_pred=y_pred_test)
                            )
                    except (KeyError, ValueError, IndexError):
                        crps_test_list.append(np.nan)

            # convert to arrays for plotting
            crps_train_arr = np.array(crps_train_list, dtype=float)
            crps_test_arr = np.array(crps_test_list, dtype=float)

            # plot train and test on separate axes (left = train, right = test)
            x_axis = np.arange(len(cv_labels))
            if not np.all(np.isnan(crps_train_arr)):
                ax_train.plot(x_axis, crps_train_arr, marker="o", color="C0")
            if not np.all(np.isnan(crps_test_arr)):
                ax_test.plot(x_axis, crps_test_arr, marker="o", color="C1")

            # Title for this pair of subplots
            title_dims = list(dims.keys() if dims else []) + additional_dims
            title_values = []
            for v in dims_combo:
                title_values.append(v)
            for k in dims or {}:
                if k not in dims_keys:
                    title_values.append((dims or {})[k])
            if addl_combo:
                title_values.extend(addl_combo)
            subplot_title = self._build_subplot_title(
                title_dims, tuple(title_values), fallback_title="CRPS per dimension"
            )

            ax_train.set_title(f"{subplot_title} — train")
            ax_test.set_title(f"{subplot_title} — test")
            ax_train.set_xlabel("Iteration")
            ax_test.set_xlabel("Iteration")
            ax_train.set_ylabel("CRPS")
            ax_test.set_ylabel("CRPS")
            ax_train.legend(["train"], loc="best")
            ax_test.legend(["test"], loc="best")

        fig.suptitle("CRPS per dimension", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        return fig, axes
