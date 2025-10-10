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
    _ = mmm.plot.posterior_predictive(var=["y"], hdi_prob=0.9)

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
from collections.abc import Iterable
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

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

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created Figure object.
        axes : np.ndarray of matplotlib.axes.Axes
            2D array of axes of shape (n_subplots, ncols).
        """
        fig, axes = plt.subplots(
            nrows=n_subplots,
            ncols=ncols,
            figsize=(width_per_col * ncols, height_per_row * n_subplots),
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

    def _add_median_and_hdi(
        self, ax: Axes, data: xr.DataArray, var: str, hdi_prob: float = 0.85
    ) -> Axes:
        """Add median and HDI to the given axis."""
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
        ax.plot(dates, median, label=var, alpha=0.9)
        ax.fill_between(dates, hdi[var][..., 0], hdi[var][..., 1], alpha=0.2)
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

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self,
        var: list[str] | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
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

    def contributions_over_time(
        self,
        var: list[str],
        hdi_prob: float = 0.85,
        dims: dict[str, str | int | list] | None = None,
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
        fig, axes = self._init_subplots(len(total_combos), ncols=1)

        for row_idx, (dims_combo, addl_combo) in enumerate(total_combos):
            ax = axes[row_idx][0]
            # Build indexers for dims and additional_dims
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

            # Plot posterior median and HDI for each var
            for v in var:
                data = self.idata.posterior[v]
                missing_coords = {
                    key: value for key, value in coords.items() if key not in data.dims
                }
                data = data.expand_dims(**missing_coords)
                data = data.sel(**indexers)  # apply slice
                data = self._reduce_and_stack(
                    data, dims_to_ignore={"date", "chain", "draw", "sample"}
                )
                ax = self._add_median_and_hdi(ax, data, v, hdi_prob=hdi_prob)

            # Title includes both fixed and combo dims
            title_dims = (
                list(dims.keys()) + additional_dims if dims else additional_dims
            )
            title_combo = tuple(indexers[k] for k in title_dims)

            title = self._build_subplot_title(
                dims=title_dims, combo=title_combo, fallback_title="Time Series"
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Value")
            ax.legend(loc="best")

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

    def allocated_contribution_by_channel_over_time(
        self,
        samples: xr.Dataset,
        scale_factor: float | None = None,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        original_scale: bool = True,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> tuple[Figure, plt.Axes | NDArray[Axes]]:
        """Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along with
        the uncertainty intervals defined by the lower and upper quantiles.
        If additional dimensions besides 'channel', 'date', and 'sample' are present,
        creates a subplot for each combination of these dimensions.

        Parameters
        ----------
        samples : xr.Dataset
            The dataset containing the samples of channel contributions.
            Expected to have 'channel_contribution' variable with dimensions
            'channel', 'date', and 'sample'.
        scale_factor : float, optional
            Scale factor to convert to original scale, if original_scale=True.
            If None and original_scale=True, assumes scale_factor=1.
        lower_quantile : float, optional
            The lower quantile for the uncertainty interval. Default is 0.025.
        upper_quantile : float, optional
            The upper quantile for the uncertainty interval. Default is 0.975.
        original_scale : bool, optional
            If True, the contributions are plotted on the original scale. Default is True.
        figsize : tuple[float, float], optional
            The size of the figure to be created. Default is (10, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
            Only used when no extra dimensions are present.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            The Axes object with the plot, or array of Axes for multiple subplots.
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

        # If no extra dimensions or using provided axis, create a single plot
        if not extra_dims or ax is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()

            channel_contribution = samples[channel_contrib_var]

            # Apply scale factor if in original scale
            if original_scale and scale_factor is not None:
                channel_contribution = channel_contribution * scale_factor

            # Plot mean values by channel
            channel_contribution.mean(dim="sample").plot(hue="channel", ax=ax)

            # Add uncertainty intervals for each channel
            for channel in samples.coords["channel"].values:
                ax.fill_between(
                    x=channel_contribution.date.values,
                    y1=channel_contribution.sel(channel=channel).quantile(
                        lower_quantile, dim="sample"
                    ),
                    y2=channel_contribution.sel(channel=channel).quantile(
                        upper_quantile, dim="sample"
                    ),
                    alpha=0.1,
                )

            ax.set_xlabel("Date")
            ax.set_ylabel("Channel Contribution")
            ax.set_title("Allocated Contribution by Channel Over Time")

            fig.tight_layout()
            return fig, ax

        # For multiple dimensions, create a grid of subplots
        # Determine layout based on number of extra dimensions
        if len(extra_dims) == 1:
            # One extra dimension: use for rows
            dim_values = [samples.coords[extra_dims[0]].values]
            nrows = len(dim_values[0])
            ncols = 1
            subplot_dims = [extra_dims[0], None]
        elif len(extra_dims) == 2:
            # Two extra dimensions: one for rows, one for columns
            dim_values = [
                samples.coords[extra_dims[0]].values,
                samples.coords[extra_dims[1]].values,
            ]
            nrows = len(dim_values[0])
            ncols = len(dim_values[1])
            subplot_dims = extra_dims
        else:
            # Three or more: use first two for rows/columns, average over the rest
            dim_values = [
                samples.coords[extra_dims[0]].values,
                samples.coords[extra_dims[1]].values,
            ]
            nrows = len(dim_values[0])
            ncols = len(dim_values[1])
            subplot_dims = [extra_dims[0], extra_dims[1]]

        # Calculate figure size based on number of subplots
        subplot_figsize = (figsize[0] * max(1, ncols), figsize[1] * max(1, nrows))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=subplot_figsize)

        # Make axes indexable even for 1x1 grid
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        # Create a subplot for each combination of dimension values
        for i, row_val in enumerate(dim_values[0]):
            for j, col_val in enumerate(
                dim_values[1] if len(dim_values) > 1 else [None]
            ):
                ax = axes[i, j]

                # Select data for this subplot
                selection = {subplot_dims[0]: row_val}
                if col_val is not None:
                    selection[subplot_dims[1]] = col_val

                # Select channel contributions for this subplot
                subset = samples[channel_contrib_var].sel(**selection)

                # Apply scale factor if needed
                if original_scale and scale_factor is not None:
                    subset = subset * scale_factor

                # Plot mean values by channel for this subset
                subset.mean(dim="sample").plot(hue="channel", ax=ax)

                # Add uncertainty intervals for each channel
                for channel in samples.coords["channel"].values:
                    channel_data = subset.sel(channel=channel)
                    ax.fill_between(
                        x=channel_data.date.values,
                        y1=channel_data.quantile(lower_quantile, dim="sample"),
                        y2=channel_data.quantile(upper_quantile, dim="sample"),
                        alpha=0.1,
                    )

                # Add subplot title based on dimension values
                title_parts = []
                if subplot_dims[0] is not None:
                    title_parts.append(f"{subplot_dims[0]}={row_val}")
                if subplot_dims[1] is not None:
                    title_parts.append(f"{subplot_dims[1]}={col_val}")

                base_title = "Allocated Contribution by Channel Over Time"
                if title_parts:
                    ax.set_title(f"{base_title} - {', '.join(title_parts)}")
                else:
                    ax.set_title(base_title)

                ax.set_xlabel("Date")
                ax.set_ylabel("Channel Contribution")

        fig.tight_layout()
        return fig, axes

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
        """
        if not hasattr(self.idata, "sensitivity_analysis"):
            raise ValueError(
                "No sensitivity analysis results found. Run run_sweep() first."
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
        # Determine plotting dimensions (excluding sample & sweep)
        plot_dims = [d for d in x.dims if d not in {"sample", "sweep"}]
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

        axes_flat = axes_array.flatten()
        for idx, combo in enumerate(dim_combinations):
            current_ax = axes_flat[idx]
            indexers = dict(zip(plot_dims, combo, strict=False)) if plot_dims else {}
            subset = x.sel(**indexers) if indexers else x
            subset = subset.squeeze(drop=True)
            subset = subset.astype(float)

            if "sweep" in subset.dims:
                sweep_dim = "sweep"
            else:
                cand = [d for d in subset.dims if d != "sample"]
                if not cand:
                    raise ValueError(
                        "Expected 'sweep' (or a non-sample) dimension in sensitivity results."
                    )
                sweep_dim = cand[0]

            sweep = (
                np.asarray(subset.coords[sweep_dim].values)
                if sweep_dim in subset.coords
                else np.arange(subset.sizes[sweep_dim])
            )

            mean = subset.mean("sample") if "sample" in subset.dims else subset
            reduce_dims = [d for d in mean.dims if d != sweep_dim]
            if reduce_dims:
                mean = mean.sum(dim=reduce_dims)

            if "sample" in subset.dims:
                hdi = az.hdi(subset, hdi_prob=hdi_prob, input_core_dims=[["sample"]])
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

            current_ax.plot(sweep, np.asarray(mean.values, dtype=float), **_plot_kwargs)
            az.plot_hdi(
                x=sweep,
                hdi_data=np.asarray(hdi.values, dtype=float),
                hdi_prob=hdi_prob,
                color=_line_color,
                ax=current_ax,
            )

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
