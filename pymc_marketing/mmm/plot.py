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

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz_base.labels import DimCoordLabeller, NoVarLabeller, mix_labellers
from arviz_plots import PlotCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

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

    def _resolve_backend(self, backend: str | None) -> str:
        """Resolve backend parameter to actual backend string."""
        from pymc_marketing.mmm.config import mmm_config

        return backend or mmm_config["plot.backend"]

    # ------------------------------------------------------------------------
    #                          Main Plotting Methods
    # ------------------------------------------------------------------------

    def posterior_predictive(
        self,
        var: list[str] | None = None,
        idata: xr.Dataset | None = None,
        hdi_prob: float = 0.85,
        backend: str | None = None,
    ) -> PlotCollection:
        """
        Plot posterior predictive distributions over time.

        Parameters
        ----------
        var : list of str, optional
            List of variable names to plot. If None, uses "y".
        idata : xr.Dataset, optional
            Dataset containing posterior predictive samples.
            If None, uses self.idata.posterior_predictive.
        hdi_prob : float, default 0.85
            Probability mass for HDI interval.
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default (via config) is "matplotlib".

        Returns
        -------
        PlotCollection

        """
        if not 0 < hdi_prob < 1:
            raise ValueError("HDI probability must be between 0 and 1.")

        # Resolve backend
        backend = self._resolve_backend(backend)

        # 1. Retrieve or validate posterior_predictive data
        pp_data = self._get_posterior_predictive_data(idata)

        # 2. Determine variables to plot
        if var is None:
            var = ["y"]
        main_var = var[0]

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
        hdi_prob: float = 0.85,
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
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
        backend : str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default (via config) is "matplotlib".

        Returns
        -------
        PlotCollection

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

        # Resolve backend
        backend = self._resolve_backend(backend)

        main_var = var[0]
        ignored_dims = {"chain", "draw", "date"}
        da = self.idata.posterior[var]
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
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot the saturation curves for each channel.

        Creates a grid of subplots for each combination of channel and non-(date/channel) dimensions.
        Optionally, subset by dims (single values or lists).
        Each channel will have a consistent color across all subplots.

        Parameters
        ----------
        original_scale: bool, optional
            Whether to plot the original scale contributions. Default is False.
        dims: dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "user_type": "new"}.
            If provided, only the selected slice(s) will be plotted.
        backend: str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default (via config) is "matplotlib".

        Returns
        -------
        PlotCollection
        """
        # Resolve backend
        backend = self._resolve_backend(backend)

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

        pc = azp.PlotCollection.grid(
            self.idata.posterior[channel_contribution]
            .mean(dim=["chain", "draw"])
            .to_dataset(),
            cols=additional_dims,
            rows=["channel"],
            aes={"color": ["channel"]},
            backend=backend,
        )
        pc.map(
            azp.visuals.scatter_xy,
            x=self.idata.constant_data.channel_data,
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
        n_samples: int = 10,
        hdi_probs: float | list[float] | None = None,
        random_seed: np.random.Generator | None = None,
        dims: dict[str, str | int | list] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
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
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "region": "X"}.
            If provided, only the selected slice(s) will be plotted.
        backend: str, optional
            Plotting backend to use. Options: "matplotlib", "plotly", "bokeh".
            If None, uses global config via mmm_config["plot.backend"].
            Default (via config) is "matplotlib".

        Returns
        -------
        PlotCollection

        Example use:
        >>> curve = model.saturation.sample_curve(
        >>>     model.idata.posterior[["saturation_beta", "saturation_lam"]], max_value=2
        >>> )
        >>> pc = model.plot.saturation_curves(curve, original_scale=True, n_samples=10,
        >>>     hdi_probs=[0.9, 0.7], random_seed=rng)
        >>> pc.show()
        """
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
        if original_scale:
            curve_data = curve * self.idata.constant_data.target_scale
            curve_data["x"] = curve_data["x"] * self.idata.constant_data.channel_scale
        else:
            curve_data = curve
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

        # create the saturation scatterplot
        pc = self.saturation_scatterplot(
            original_scale=original_scale, dims=dims, backend=backend
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

    def saturation_curves_scatter(
        self, original_scale: bool = False, **kwargs
    ) -> PlotCollection:
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
        PlotCollection
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

    def budget_allocation_roas(
        self,
        samples: xr.Dataset,
        dims: dict[str, str | int | list] | None = None,
        dims_to_group_by: list[str] | str | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot the ROI distribution of a given a response distribution and a budget allocation.

        Parameters
        ----------
        samples : xr.Dataset
            The dataset containing the channel contributions and allocation values.
            Expected to have 'channel_contribution' and 'allocation' variables.
        dims : dict[str, str | int | list], optional
            Dimension filters to apply. Example: {"country": ["US", "UK"], "user_type": "new"}.
            If provided, only the selected slice(s) will be plotted.
        dims_to_group_by : list[str] | str | None, optional
            Dimension(s) to group by for plotting purposes.
            When a dimension is specified, all the ROAs distributions for each coordinate of that dimension will be
            plotted together in a single plot. This is useful for comparing the ROAs distributions.
            If None, will not group by any dimensions (i.e. each distribution will be plotted separately).
            If a single string, will group by that dimension.
            If a list of strings, will group by each of those dimensions.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

        Returns
        -------
        PlotCollection
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
            for k, v in grouped.items():
                grouped_roa_dt[k[5:]] = v
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
        hdi_prob : float, optional
            The probability mass of the highest density interval to be displayed. Default is 0.85.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

        Returns
        -------
        PlotCollection
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
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot helper for sensitivity analysis results.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

        Returns
        -------
        PlotCollection

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
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """Plot sensitivity analysis results.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

        Returns
        -------
        PlotCollection

        Examples
        --------
        Basic run using stored results in `idata`:

        .. code-block:: python

            # Assuming you already ran a sweep and stored results
            # under idata.sensitivity_analysis via SensitivityAnalysis.run_sweep(..., extend_idata=True)
            mmm.plot.sensitivity_analysis(hdi_prob=0.9)

        With aggregation over dimensions (e.g., sum over channels):

        .. code-block:: python

            mmm.plot.sensitivity_analysis(
                hdi_prob=0.9,
                aggregation={"sum": ("channel",)},
            )
        """
        pc = self._sensitivity_analysis_plot(
            hdi_prob=hdi_prob, aggregation=aggregation, backend=backend
        )
        pc.map(azp.visuals.labelled_y, text="Contribution")
        return pc

    def uplift_curve(
        self,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """
        Plot precomputed uplift curves stored under `idata.sensitivity_analysis['uplift_curve']`.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

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
            mmm.plot.uplift_curve(hdi_prob=0.9)
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
            pc = self._sensitivity_analysis_plot(
                hdi_prob=hdi_prob,
                aggregation=aggregation,
                backend=backend,
            )
            pc.map(azp.visuals.labelled_y, text="Uplift (%)")
            return pc
        finally:
            self.idata.sensitivity_analysis = original_group  # type: ignore

    def marginal_curve(
        self,
        hdi_prob: float = 0.94,
        aggregation: dict[str, tuple[str, ...] | list[str]] | None = None,
        backend: str | None = None,
    ) -> PlotCollection:
        """
        Plot precomputed marginal effects stored under `idata.sensitivity_analysis['marginal_effects']`.

        Parameters
        ----------
        hdi_prob : float, default 0.94
            HDI probability mass.
        aggregation : dict, optional
            Aggregation to apply to the data.
            E.g., {"sum": ("channel",)} to sum over the channel dimension.
        backend : str | None, optional
            Backend to use for plotting. If None, will use the global backend configuration.

        Returns
        -------
        PlotCollection

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
            mmm.plot.marginal_curve(hdi_prob=0.9)
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
            pc = self._sensitivity_analysis_plot(
                hdi_prob=hdi_prob,
                aggregation=aggregation,
                backend=backend,
            )
            pc.map(azp.visuals.labelled_y, text="Marginal Effect")
            return pc
        finally:
            self.idata.sensitivity_analysis = original  # type: ignore
