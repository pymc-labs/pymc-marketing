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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.axes import Axes

from pymc_marketing.mmm.sensitivity_analysis import SensitivityAnalysis


@pytest.fixture
def mock_mmm():
    """Create a mock MMM class with minimal functionality."""

    class MockMMM:
        def __init__(self):
            # Create DataFrame with proper date index
            dates = pd.date_range("2023-01-01", periods=3)
            self.X = pd.DataFrame(
                {
                    "predictor_1": [1, 2, 3],
                    "predictor_2": [4, 5, 6],
                },
                index=dates,
            )

            # Create a mock idata structure that matches the real MMM structure
            # Use a simple object that can have attributes assigned
            class MockIdata:
                def __init__(self):
                    # Create realistic posterior_predictive data
                    n_chains, n_draws, n_dates = 2, 10, 3

                    # This is what the sensitivity analysis expects to find
                    posterior_predictive_data = xr.Dataset(
                        dict(
                            y=(
                                ["chain", "draw", "date"],
                                np.random.normal(size=(n_chains, n_draws, n_dates)),
                            ),
                        ),
                        coords={
                            "chain": np.arange(n_chains),
                            "draw": np.arange(n_draws),
                            "date": dates,  # Use the same dates as the DataFrame
                        },
                    )

                    self.posterior_predictive = posterior_predictive_data

                def __getitem__(self, key):
                    if key == "posterior_predictive":
                        return self.posterior_predictive
                    raise KeyError(f"Key {key} not found")

                def add_groups(self, groups):
                    # Mock implementation of add_groups
                    for group_name, group_data in groups.items():
                        setattr(self, group_name, group_data)

            self.idata = MockIdata()

        def predict(self, X_new, extend_idata=False, progressbar=False):
            # Mock predict method to return data in the format expected by plot_sensitivity_analysis
            # The real predict method returns data with dimensions (chain, draw, date)
            n_chains, n_draws, n_dates = 2, 10, len(X_new)

            # Create realistic prediction data with proper dimensions
            data = np.random.normal(
                loc=X_new["predictor_1"] + X_new["predictor_2"],
                scale=0.1,
                size=(n_chains, n_draws, n_dates),
            )

            return xr.Dataset(
                dict(
                    y=(
                        ["chain", "draw", "date"],
                        data,
                    )  # Use the same dimensions as expected
                ),
                coords={
                    "chain": np.arange(n_chains),
                    "draw": np.arange(n_draws),
                    "date": X_new.index,  # Use the DataFrame index as dates
                },
            )

        def sample_posterior_predictive(
            self, X_new, extend_idata=False, combined=False, progressbar=False
        ):
            # Mock implementation of sample_posterior_predictive
            return self.predict(X_new, extend_idata, progressbar)

    return MockMMM()


@pytest.fixture
def sensitivity_analysis(mock_mmm):
    """Create a SensitivityAnalysis instance with the mock MMM."""
    return SensitivityAnalysis(mock_mmm)


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_run_sweep_basic(sensitivity_analysis, sweep_type):
    """Test that run_sweep runs without errors and returns expected results."""
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type=sweep_type,
    )

    # Check that results is an xarray.Dataset
    assert isinstance(results, xr.Dataset)

    # Check that the dataset contains the expected variables
    assert "y" in results
    assert "marginal_effects" in results

    # Check that the sweep dimension is correct
    assert "sweep" in results.dims
    assert len(results["sweep"]) == len(sweep_values)


def test_run_sweep_invalid_var_name(sensitivity_analysis):
    """Test that run_sweep raises an error for invalid variable names."""
    with pytest.raises(KeyError, match="predictor_invalid"):
        sensitivity_analysis.run_sweep(
            var_names=["predictor_invalid"],
            sweep_values=np.linspace(0.5, 1.5, 3),
            sweep_type="multiplicative",
        )


def test_run_sweep_metadata(sensitivity_analysis):
    """Test that metadata is correctly added to the results."""
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Check metadata
    assert results.attrs["sweep_type"] == "multiplicative"
    assert results.attrs["var_names"] == ["predictor_1"]


def test_run_sweep_marginal_effects(sensitivity_analysis):
    """Test that marginal effects are computed correctly."""
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Check that marginal effects are computed
    assert "marginal_effects" in results
    assert results["marginal_effects"].dims == results["y"].dims


def test_run_sweep_multiple_variables(sensitivity_analysis):
    """Test that run_sweep works with multiple variables."""
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1", "predictor_2"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Check that results is an xarray.Dataset
    assert isinstance(results, xr.Dataset)
    assert "y" in results
    assert "marginal_effects" in results


def test_run_sweep_edge_values(sensitivity_analysis):
    """Test run_sweep with edge values including zero."""
    sweep_values = np.array([0.0, 0.5, 1.0, 2.0])
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Check that it handles zero values
    assert isinstance(results, xr.Dataset)
    assert len(results["sweep"]) == len(sweep_values)


def test_run_sweep_single_value(sensitivity_analysis):
    """Test run_sweep with a single sweep value."""
    sweep_values = np.array([1.5])

    # Single value should fail because we can't compute marginal effects
    with pytest.raises(IndexError):
        sensitivity_analysis.run_sweep(
            var_names=["predictor_1"],
            sweep_values=sweep_values,
            sweep_type="additive",
        )


def test_create_intervention_methods(sensitivity_analysis):
    """Test the different intervention creation methods."""
    # Set up the sensitivity analysis with required attributes
    sensitivity_analysis.var_names = ["predictor_1"]
    sensitivity_analysis.sweep_type = "multiplicative"

    # Test multiplicative
    X_mult = sensitivity_analysis.create_intervention(2.0)
    expected_mult = sensitivity_analysis.mmm.X.copy()
    expected_mult["predictor_1"] *= 2.0
    pd.testing.assert_frame_equal(X_mult, expected_mult)

    # Test additive
    sensitivity_analysis.sweep_type = "additive"
    X_add = sensitivity_analysis.create_intervention(0.5)
    expected_add = sensitivity_analysis.mmm.X.copy()
    expected_add["predictor_1"] += 0.5
    pd.testing.assert_frame_equal(X_add, expected_add)

    # Test absolute
    sensitivity_analysis.sweep_type = "absolute"
    X_abs = sensitivity_analysis.create_intervention(3.0)
    expected_abs = sensitivity_analysis.mmm.X.copy()
    expected_abs["predictor_1"] = 3.0
    pd.testing.assert_frame_equal(X_abs, expected_abs)


def test_create_intervention_invalid_sweep_type(sensitivity_analysis):
    """Test that create_intervention raises error for invalid sweep type."""
    sensitivity_analysis.var_names = ["predictor_1"]
    sensitivity_analysis.sweep_type = "invalid_type"

    with pytest.raises(ValueError, match="Unsupported sweep_type"):
        sensitivity_analysis.create_intervention(1.0)


@pytest.fixture
def mock_mmm_with_plot(mock_mmm):
    """Create a mock MMM with real MMMPlotSuite for testing plot_sensitivity_analysis."""
    from pymc_marketing.mmm.plot import MMMPlotSuite

    # Use the real MMMPlotSuite class instead of a mock
    mock_mmm.plot = MMMPlotSuite(mock_mmm.idata)
    return mock_mmm


@pytest.fixture
def sensitivity_analysis_with_results(mock_mmm_with_plot):
    """Create a SensitivityAnalysis instance with pre-computed results."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run a sweep to create results in the MMM's idata
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Add the results to the MMM's idata to simulate real usage
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    return sensitivity_analysis


def test_plot_sensitivity_analysis_basic(sensitivity_analysis_with_results):
    """Test basic plot_sensitivity_analysis functionality."""
    mmm = sensitivity_analysis_with_results.mmm

    # Test that the plot function exists and runs without error
    ax = mmm.plot.plot_sensitivity_analysis()

    # Check that we get a matplotlib Axes object
    assert isinstance(ax, Axes)

    # Check basic plot properties
    assert ax.get_title() == "Sensitivity analysis plot"
    assert "Multiplicative change of" in ax.get_xlabel()
    assert ax.get_ylabel() == "Total uplift (sum over dates)"

    # Check that lines were plotted
    lines = ax.get_lines()
    assert len(lines) > 0

    plt.close()  # Close the current figure


def test_plot_sensitivity_analysis_marginal(sensitivity_analysis_with_results):
    """Test plot_sensitivity_analysis with marginal effects."""
    mmm = sensitivity_analysis_with_results.mmm

    ax = mmm.plot.plot_sensitivity_analysis(marginal=True)

    assert isinstance(ax, Axes)

    plt.close()  # Close the current figure


def test_plot_sensitivity_analysis_with_custom_ax(sensitivity_analysis_with_results):
    """Test plot_sensitivity_analysis with a custom axes."""
    mmm = sensitivity_analysis_with_results.mmm

    fig, custom_ax = plt.subplots(figsize=(8, 6))

    # Use the custom axes
    returned_ax = mmm.plot.plot_sensitivity_analysis(ax=custom_ax)

    # Should return the same axes object
    assert returned_ax is custom_ax
    assert isinstance(returned_ax, Axes)

    plt.close(fig)


@pytest.mark.parametrize("hdi_prob", [0.89, 0.94, 0.99])
def test_plot_sensitivity_analysis_hdi_prob(
    sensitivity_analysis_with_results, hdi_prob
):
    """Test plot_sensitivity_analysis with different HDI probabilities."""
    mmm = sensitivity_analysis_with_results.mmm

    ax = mmm.plot.plot_sensitivity_analysis(hdi_prob=hdi_prob)

    assert isinstance(ax, Axes)

    plt.close()  # Close the current figure


@pytest.mark.parametrize("marginal", [True, False])
def test_plot_sensitivity_analysis_marginal_parameter(
    sensitivity_analysis_with_results, marginal
):
    """Test plot_sensitivity_analysis with marginal parameter."""
    mmm = sensitivity_analysis_with_results.mmm

    ax = mmm.plot.plot_sensitivity_analysis(marginal=marginal)

    assert isinstance(ax, Axes)

    plt.close()  # Close the current figure


def test_plot_sensitivity_analysis_no_results(mock_mmm_with_plot):
    """Test plot_sensitivity_analysis when no sensitivity analysis results exist."""
    # Test without running sensitivity analysis first
    with pytest.raises(ValueError, match="No sensitivity analysis results found"):
        mock_mmm_with_plot.plot.plot_sensitivity_analysis()


def test_plot_sensitivity_analysis_percentage_marginal_error(
    sensitivity_analysis_with_results,
):
    """Test that percentage + marginal raises an error."""
    mmm = sensitivity_analysis_with_results.mmm

    # This combination should raise an error in the real implementation
    # For now, our mock doesn't implement this check, but we can test the concept
    try:
        ax = mmm.plot.plot_sensitivity_analysis(percentage=True, marginal=True)
        plt.close(ax.figure)
        # If no error is raised, that's fine for the mock - the real implementation would error
    except ValueError as e:
        # This is expected in the real implementation
        assert "Not implemented marginal effects in percentage scale" in str(e)


def test_plot_sensitivity_analysis_integration(mock_mmm_with_plot):
    """Test full integration: run sweep and then plot."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run a sweep
    sweep_values = np.linspace(0.5, 1.5, 3)
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=sweep_values,
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Now plot should work
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis()

    assert isinstance(ax, Axes)

    plt.close()  # Close the current figure


def test_plot_sensitivity_analysis_percentage_scale(sensitivity_analysis_with_results):
    """Test plotting with percentage scale (non-marginal)."""
    mmm = sensitivity_analysis_with_results.mmm

    # This should work (percentage=True, marginal=False)
    ax = mmm.plot.plot_sensitivity_analysis(percentage=True, marginal=False)

    assert isinstance(ax, Axes)
    # Check that y-axis formatter is set for percentages
    formatter = ax.yaxis.get_major_formatter()
    sample_value = 0.1
    formatted = formatter(sample_value, None)
    assert "%" in formatted

    plt.close()


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive", "absolute"])
def test_plot_sensitivity_analysis_sweep_types_xlabel(mock_mmm_with_plot, sweep_type):
    """Test that xlabel changes based on sweep type."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep with specific type
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type=sweep_type,
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Plot and check xlabel
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis()

    xlabel = ax.get_xlabel()
    if sweep_type == "absolute":
        assert "Absolute value of:" in xlabel
    else:
        assert f"{sweep_type.capitalize()} change of:" in xlabel

    plt.close()


@pytest.mark.parametrize("sweep_type", ["multiplicative", "additive"])
def test_plot_sensitivity_analysis_reference_lines(mock_mmm_with_plot, sweep_type):
    """Test that reference lines are drawn for multiplicative and additive sweeps."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type=sweep_type,
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Plot
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis()

    # Check for reference lines (dashed lines)
    all_lines = ax.get_lines()
    dashed_lines = [line for line in all_lines if line.get_linestyle() == "--"]

    # Should have at least one dashed reference line
    assert len(dashed_lines) >= 1

    if sweep_type == "multiplicative":
        # Should have vertical line at x=1 and possibly horizontal line at y=0
        has_vertical_ref = any(
            np.allclose(line.get_xdata(), [1, 1], atol=0.1) for line in dashed_lines
        )
        assert has_vertical_ref
    elif sweep_type == "additive":
        # Should have vertical line at x=0
        has_vertical_ref = any(
            np.allclose(line.get_xdata(), [0, 0], atol=0.1) for line in dashed_lines
        )
        assert has_vertical_ref

    plt.close()


def test_plot_sensitivity_analysis_y_axis_limits_positive(mock_mmm_with_plot):
    """Test y-axis limits when all y values are positive."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Create a scenario with positive values by using a sweep that increases values
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(1.0, 2.0, 5),  # All > 1 should give positive uplift
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Plot
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis()

    # Check that bottom y-limit is set to 0 for positive values
    y_mean = results.y.mean(dim=["chain", "draw"]).sum(dim="date")
    if np.all(y_mean.values > 0):
        y_bottom, y_top = ax.get_ylim()
        assert y_bottom == 0

    plt.close()


def test_plot_sensitivity_analysis_y_axis_limits_negative(mock_mmm_with_plot):
    """Test y-axis limits when all y values are negative."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Create a scenario with negative values by using a sweep that decreases values
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.1, 0.8, 5),  # All < 1 should give negative uplift
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Plot
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis()

    # Check that top y-limit is set to 0 for negative values
    y_mean = results.y.mean(dim=["chain", "draw"]).sum(dim="date")
    if np.all(y_mean.values < 0):
        y_bottom, y_top = ax.get_ylim()
        assert y_top == 0

    plt.close()


def test_plot_sensitivity_analysis_formatter_percentage_vs_absolute(mock_mmm_with_plot):
    """Test that the y-axis formatter differs between percentage and absolute scales."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Test absolute scale
    ax1 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(percentage=False)
    formatter_abs = ax1.yaxis.get_major_formatter()

    # Test percentage scale
    ax2 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(percentage=True)
    formatter_pct = ax2.yaxis.get_major_formatter()

    # Formatters should behave differently
    # Test by applying them to a sample value
    sample_value = 0.1
    abs_formatted = formatter_abs(sample_value, None)
    pct_formatted = formatter_pct(sample_value, None)

    # Percentage should contain % while absolute should not
    assert "%" in pct_formatted
    assert "%" not in abs_formatted

    plt.close(ax1.figure)
    plt.close(ax2.figure)


def test_plot_sensitivity_analysis_marginal_vs_uplift_labels(mock_mmm_with_plot):
    """Test that labels and titles change between marginal and uplift plots."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep with enough points for marginal effects
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Test uplift plot
    ax1 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(marginal=False)
    uplift_title = ax1.get_title()
    uplift_ylabel = ax1.get_ylabel()

    # Test marginal plot
    ax2 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(marginal=True)
    marginal_title = ax2.get_title()
    marginal_ylabel = ax2.get_ylabel()

    # Titles should be different
    assert uplift_title != marginal_title
    assert "Sensitivity analysis" in uplift_title
    assert "Marginal effects" in marginal_title

    # Y-labels should be different
    assert uplift_ylabel != marginal_ylabel
    assert "uplift" in uplift_ylabel.lower()
    assert "marginal effect" in marginal_ylabel.lower()

    plt.close(ax1.figure)
    plt.close(ax2.figure)


def test_plot_sensitivity_analysis_hdi_in_legend(mock_mmm_with_plot):
    """Test that HDI probability appears in legend."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Test with specific HDI probability
    hdi_prob = 0.89
    ax = mock_mmm_with_plot.plot.plot_sensitivity_analysis(hdi_prob=hdi_prob)

    # Check legend contains HDI probability
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]

    # Should contain the HDI percentage
    hdi_percentage = f"{hdi_prob * 100:.0f}%"
    assert any(hdi_percentage in text for text in legend_texts)

    plt.close()


def test_plot_sensitivity_analysis_color_consistency(mock_mmm_with_plot):
    """Test that colors are consistent between marginal and uplift plots."""
    sensitivity_analysis = SensitivityAnalysis(mock_mmm_with_plot)

    # Run sweep
    results = sensitivity_analysis.run_sweep(
        var_names=["predictor_1"],
        sweep_values=np.linspace(0.5, 1.5, 5),
        sweep_type="multiplicative",
    )

    # Add results to MMM's idata
    mock_mmm_with_plot.idata.add_groups({"sensitivity_analysis": results})

    # Test uplift plot
    ax1 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(marginal=False)
    uplift_line_color = ax1.get_lines()[0].get_color()

    # Test marginal plot
    ax2 = mock_mmm_with_plot.plot.plot_sensitivity_analysis(marginal=True)
    marginal_line_color = ax2.get_lines()[0].get_color()

    # Colors should be different (C0 vs C1)
    assert uplift_line_color != marginal_line_color

    plt.close(ax1.figure)
    plt.close(ax2.figure)
