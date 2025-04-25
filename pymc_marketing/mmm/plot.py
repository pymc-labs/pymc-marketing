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
"""MMM related plotting class."""

import itertools

import arviz as az
import matplotlib.pyplot as plt
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

    def _compute_ci(
        self, data: xr.DataArray, ci: float = 0.85, sample_dim: str = "sample"
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Compute median and lower/upper credible intervals over given sample_dim."""
        lower_q = 0.5 - ci / 2
        upper_q = 0.5 + ci / 2
        data_median = data.quantile(0.5, dim=sample_dim)
        data_lower = data.quantile(lower_q, dim=sample_dim)
        data_upper = data.quantile(upper_q, dim=sample_dim)
        return data_median, data_lower, data_upper

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

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self,
        var: list[str] | None = None,
        idata: xr.Dataset | None = None,
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
        """
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
                # Compute median & 85% intervals
                median, lower, upper = self._compute_ci(data, ci=0.85)

                # Extract date coordinate
                if "date" not in data.dims:
                    raise ValueError(
                        f"Expected 'date' dimension in {v}, but none found."
                    )
                dates = data.coords["date"].values

                # Plot
                ax.plot(dates, median, label=v, alpha=0.9)
                ax.fill_between(dates, lower, upper, alpha=0.2)

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
        ci: float = 0.85,
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the time-series contributions for each variable in `var`.

        showing the median and the credible interval (default 85%).
        Creates one subplot per combination of non-(chain/draw/date) dimensions
        and places all variables on the same subplot.

        Parameters
        ----------
        var : list of str
            A list of variable names to plot from the posterior.
        ci : float, optional
            Credible interval width. For instance, 0.85 will show the
            7.5th to 92.5th percentile range. The default is 0.85.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the subplots.
        axes : np.ndarray of matplotlib.axes.Axes
            Array of Axes objects corresponding to each subplot row.
        """
        if not 0 < ci < 1:
            raise ValueError("Credible interval must be between 0 and 1.")

        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "No posterior data found in 'self.idata'. "
                "Please ensure 'self.idata' contains a 'posterior' group."
            )

        main_var = var[0]
        all_dims = list(self.idata.posterior[main_var].dims)  # type: ignore
        ignored_dims = {"chain", "draw", "date"}
        additional_dims = [d for d in all_dims if d not in ignored_dims]

        # Identify combos
        if additional_dims:
            additional_coords = [
                self.idata.posterior.coords[dim].values  # type: ignore
                for dim in additional_dims  # type: ignore
            ]
            dim_combinations = list(itertools.product(*additional_coords))
        else:
            dim_combinations = [()]

        # Prepare subplots
        fig, axes = self._init_subplots(len(dim_combinations), ncols=1)

        # Loop combos
        for row_idx, combo in enumerate(dim_combinations):
            ax = axes[row_idx][0]
            indexers = (
                dict(zip(additional_dims, combo, strict=False))
                if additional_dims
                else {}
            )

            # Plot each var
            for v in var:
                data = self.idata.posterior[v].sel(**indexers)  # type: ignore
                data = self._reduce_and_stack(
                    data, dims_to_ignore={"date", "chain", "draw", "sample"}
                )

                # Compute median and credible intervals
                median, lower, upper = self._compute_ci(data, ci=ci)

                # Extract dates
                dates = data.coords["date"].values
                ax.plot(dates, median, label=f"{v}", alpha=0.9)
                ax.fill_between(dates, lower, upper, alpha=0.2)

            title = self._build_subplot_title(
                dims=additional_dims, combo=combo, fallback_title="Time Series"
            )
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Posterior Value")
            ax.legend(loc="best")

        return fig, axes

    def saturation_curves_scatter(
        self, original_scale: bool = False, **kwargs
    ) -> tuple[Figure, NDArray[Axes]]:
        """Plot the saturation curves for each channel.

        Creates one subplot per combination of non-(date/channel) dimensions
        and places all channels on the same subplot.
        """
        if not hasattr(self.idata, "constant_data"):
            raise ValueError(
                "No 'constant_data' found in 'self.idata'. "
                "Please ensure 'self.idata' contains the constant_data group."
            )

        # Identify additional dimensions beyond 'date' and 'channel'
        cdims = self.idata.constant_data.channel_data.dims
        additional_dims = [dim for dim in cdims if dim not in ("date", "channel")]

        # Get all possible combinations
        if additional_dims:
            additional_coords = [
                self.idata.constant_data.coords[d].values for d in additional_dims
            ]
            additional_combinations = list(itertools.product(*additional_coords))
        else:
            additional_combinations = [()]

        # Channel in original_scale if selected
        channel_contribution = (
            "channel_contribution_original_scale"
            if original_scale
            else "channel_contribution"
        )

        if original_scale and not hasattr(self.idata.posterior, channel_contribution):
            raise ValueError(
                f"""No posterior.{channel_contribution} data found in 'self.idata'.
                Add a original scale deterministic:
                    mmm.add_original_scale_contribution_variable(
                        var=[
                            "channel_contribution",
                            ...
                        ]
                    )
                """
            )

        # Rows = channels, Columns = additional_combinations
        channels = self.idata.constant_data.coords["channel"].values
        n_rows = len(channels)
        n_columns = len(additional_combinations)

        # Create subplots
        fig, axes = self._init_subplots(n_subplots=n_rows, ncols=n_columns, **kwargs)

        # Loop channels & combos
        for row_idx, channel in enumerate(channels):
            for col_idx, combo in enumerate(additional_combinations):
                ax = axes[row_idx][col_idx] if n_columns > 1 else axes[row_idx][0]
                indexers = dict(zip(additional_dims, combo, strict=False))
                indexers["channel"] = channel

                # Select X data (constant_data)
                x_data = self.idata.constant_data.channel_data.sel(**indexers)
                # Select Y data (posterior contributions) and scale if needed
                y_data = self.idata.posterior[channel_contribution].sel(**indexers)

                # Flatten chain & draw by taking mean (or sum, up to design)
                y_data = y_data.mean(dim=["chain", "draw"])

                # Ensure X and Y have matching date coords
                x_data = x_data.broadcast_like(y_data)
                y_data = y_data.broadcast_like(x_data)

                # Scatter
                ax.scatter(
                    x_data.values.flatten(),
                    y_data.values.flatten(),
                    alpha=0.8,
                    color=f"C{row_idx}",
                )

                title = self._build_subplot_title(
                    dims=["channel", *additional_dims],
                    combo=(channel, *combo),
                    fallback_title="Channel Saturation Curves",
                )
                ax.set_title(title)
                ax.set_xlabel("Channel Data (X)")
                ax.set_ylabel("Channel Contributions (Y)")

        return fig, axes

    def budget_allocation(
        self,
        samples: xr.Dataset,
        scale_factor: float | None = None,
        figsize: tuple[float, float] = (12, 6),
        ax: plt.Axes | None = None,
        original_scale: bool = True,
    ) -> tuple[Figure, plt.Axes]:
        """Plot the budget allocation and channel contributions.

        Creates a bar chart comparing allocated spend and channel contributions
        for each channel.

        Parameters
        ----------
        samples : xr.Dataset
            The dataset containing the channel contributions and allocation values.
            Expected to have 'channel_contributions' and 'allocation' variables.
        scale_factor : float, optional
            Scale factor to convert to original scale, if original_scale=True.
            If None and original_scale=True, assumes scale_factor=1.
        figsize : tuple[float, float], optional
            The size of the figure to be created. Default is (12, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
        original_scale : bool, optional
            A boolean flag to determine if the values should be plotted in their
            original scale. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
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

        # Get channel contributions, averaging over date and sample
        # Find the variable containing 'channel_contribution' in its name
        channel_contrib_var = next(
            var_name
            for var_name in samples.data_vars
            if "channel_contribution" in var_name
        )
        channel_contributions = (
            samples[channel_contrib_var].mean(dim=["date", "sample"]).to_numpy()
        )

        # Apply scale factor if in original scale
        if original_scale and scale_factor is not None:
            channel_contributions *= scale_factor

        # Get allocated spend
        allocated_spend = samples.allocation.to_numpy()

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        bar_width = 0.35
        opacity = 0.7

        # Get channel index and names
        channels = samples.coords["channel"].values
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
            channel_contributions,
            bar_width,
            color="C1",
            alpha=opacity,
            label="Channel Contributions",
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

        fig.tight_layout()
        return fig, ax

    def allocated_contribution_by_channel(
        self,
        samples: xr.Dataset,
        scale_factor: float | None = None,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        original_scale: bool = True,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
    ) -> tuple[Figure, plt.Axes]:
        """Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along with
        the uncertainty intervals defined by the lower and upper quantiles.

        Parameters
        ----------
        samples : xr.Dataset
            The dataset containing the samples of channel contributions.
            Expected to have 'channel_contributions' variable with dimensions
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

        Returns
        -------
        fig : matplotlib.figure.Figure
            The Figure object containing the plot.
        ax : matplotlib.axes.Axes
            The Axes object with the plot.
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
        channel_contributions = samples[channel_contrib_var]

        # Apply scale factor if in original scale
        if original_scale and scale_factor is not None:
            channel_contributions = channel_contributions * scale_factor

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Plot mean values by channel
        channel_contributions.mean(dim="sample").plot(hue="channel", ax=ax)

        # Add uncertainty intervals for each channel
        for channel in samples.coords["channel"].values:
            ax.fill_between(
                x=channel_contributions.date.values,
                y1=channel_contributions.sel(channel=channel).quantile(
                    lower_quantile, dim="sample"
                ),
                y2=channel_contributions.sel(channel=channel).quantile(
                    upper_quantile, dim="sample"
                ),
                alpha=0.1,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Channel Contribution")
        ax.set_title("Allocated Contribution by Channel Over Time")

        fig.tight_layout()
        return fig, ax
