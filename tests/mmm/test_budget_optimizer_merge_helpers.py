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

"""Tests for the standalone merge helpers and the legacy-wrapper dispatch."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm.budget_optimizer import (
    BudgetOptimizer,
    BuildMergedModel,
    CustomModelWrapper,
    OptimizerCompatibleModel,
    _concrete_method,
    merge_inference_data,
    merge_models_and_idata,
)

CHANNELS = ["tv", "search"]


def build_idata(
    n_draws: int = 4,
    *,
    with_constant_data: bool = True,
    beta_name: str = "beta",
) -> xr.DataTree:
    """Build a minimal posterior DataTree with an optional shared channel_data."""
    coords = {
        "chain": [0],
        "draw": list(range(n_draws)),
        "channel": CHANNELS,
        "date": [0, 1],
    }
    posterior = xr.Dataset(
        {
            beta_name: xr.DataArray(
                np.ones((1, n_draws, len(CHANNELS))),
                dims=["chain", "draw", "channel"],
                coords={k: coords[k] for k in ("chain", "draw", "channel")},
            ),
            "channel_contribution": xr.DataArray(
                np.ones((1, n_draws, len(CHANNELS), 2)),
                dims=["chain", "draw", "channel", "date"],
                coords=coords,
            ),
        }
    )
    groups: dict[str, xr.Dataset] = {"/posterior": posterior}
    if with_constant_data:
        groups["/constant_data"] = xr.Dataset(
            {
                "channel_data": xr.DataArray(
                    np.zeros((2, len(CHANNELS))),
                    dims=["date", "channel"],
                    coords={"date": [0, 1], "channel": CHANNELS},
                )
            }
        )
    return xr.DataTree.from_dict(groups)


def build_pymc_model(n_periods: int = 6) -> pm.Model:
    """Build a tiny PyMC model carrying a shared ``channel_data`` variable."""
    coords = {"date": np.arange(n_periods), "channel": CHANNELS}
    with pm.Model(coords=coords) as model:
        channel_data = pm.Data(
            "channel_data",
            np.zeros((n_periods, len(CHANNELS))),
            dims=("date", "channel"),
        )
        beta = pm.Normal("beta", mu=0, sigma=1, dims="channel")
        mu = (channel_data * beta).sum(axis=-1)
        pm.Normal("y", mu=mu, sigma=1, observed=np.zeros(n_periods))
    return model


class TestMergeInferenceData:
    """Coverage for :func:`merge_inference_data`."""

    def test_auto_generated_prefixes(self):
        merged = merge_inference_data([build_idata(), build_idata()])

        posterior = merged["posterior"].dataset
        assert "model1_beta" in posterior.data_vars
        assert "model2_beta" in posterior.data_vars
        assert "beta" not in posterior.data_vars

    def test_explicit_prefixes(self):
        merged = merge_inference_data(
            [build_idata(), build_idata()], prefixes=["north", "south"]
        )

        posterior = merged["posterior"].dataset
        assert {"north_beta", "south_beta"} <= set(posterior.data_vars)

    def test_prefix_length_mismatch_raises(self):
        with pytest.raises(
            ValueError, match=r"Number of prefixes \(1\) must match number of idatas"
        ):
            merge_inference_data([build_idata(), build_idata()], prefixes=["only-one"])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="Need at least 1 InferenceData"):
            merge_inference_data([])

    @pytest.mark.parametrize("use_every_n_draw", [1, 2, 4], ids=str)
    def test_thinning(self, use_every_n_draw):
        merged = merge_inference_data(
            [build_idata(n_draws=8), build_idata(n_draws=8)],
            prefixes=["a", "b"],
            use_every_n_draw=use_every_n_draw,
        )

        assert merged["posterior"].sizes["draw"] == 8 // use_every_n_draw

    def test_merge_on_stays_unprefixed(self):
        merged = merge_inference_data(
            [build_idata(), build_idata()],
            prefixes=["a", "b"],
            merge_on="channel_data",
        )

        constant_data = merged["constant_data"].dataset
        assert "channel_data" in constant_data.data_vars
        # The dims of the shared variable are shared too.
        assert set(constant_data["channel_data"].dims) == {"date", "channel"}

    def test_merge_on_none_prefixes_everything(self):
        merged = merge_inference_data(
            [build_idata(), build_idata()],
            prefixes=["a", "b"],
            merge_on=None,
        )

        constant_data = merged["constant_data"].dataset
        assert "channel_data" not in constant_data.data_vars
        assert {"a_channel_data", "b_channel_data"} <= set(constant_data.data_vars)

    def test_single_idata_is_allowed(self):
        """Unlike ``merge_models_and_idata``, a single idata is accepted."""
        merged = merge_inference_data([build_idata()], prefixes=["solo"])

        assert "solo_beta" in merged["posterior"].dataset.data_vars

    def test_warns_when_merge_on_absent_from_constant_data(self):
        """The dims of ``merge_on`` cannot be identified, so say so out loud."""
        with pytest.warns(UserWarning, match="was not found in the 'constant_data'"):
            merge_inference_data(
                [
                    build_idata(with_constant_data=False),
                    build_idata(with_constant_data=False),
                ],
                prefixes=["a", "b"],
            )


class TestMergeModelsAndIdata:
    """Coverage for :func:`merge_models_and_idata`."""

    def test_returns_merged_model_and_idata(self):
        merged_model, merged_idata = merge_models_and_idata(
            models=[build_pymc_model(), build_pymc_model()],
            idatas=[build_idata(), build_idata()],
            prefixes=["north", "south"],
            merge_on="channel_data",
        )

        assert isinstance(merged_model, pm.Model)
        assert "channel_data" in merged_model.named_vars
        assert {"north_beta", "south_beta"} <= set(merged_model.named_vars)
        assert {"north_beta", "south_beta"} <= set(
            merged_idata["posterior"].dataset.data_vars
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"len\(models\) \(2\) must equal"):
            merge_models_and_idata(
                models=[build_pymc_model(), build_pymc_model()],
                idatas=[build_idata()],
            )

    def test_single_model_raises(self):
        """``merge_models`` needs at least two models; the idata side allows one."""
        with pytest.raises(ValueError, match="Need at least 2 models to merge"):
            merge_models_and_idata(
                models=[build_pymc_model()],
                idatas=[build_idata()],
            )

    def test_thinning_is_forwarded(self):
        _merged_model, merged_idata = merge_models_and_idata(
            models=[build_pymc_model(), build_pymc_model()],
            idatas=[build_idata(n_draws=8), build_idata(n_draws=8)],
            prefixes=["a", "b"],
            use_every_n_draw=4,
        )

        assert merged_idata["posterior"].sizes["draw"] == 2


class TestDeprecationWarnings:
    """Each deprecated entry point must keep announcing its replacement."""

    def test_custom_model_wrapper_warns(self):
        with pytest.warns(DeprecationWarning, match="CustomModelWrapper is deprecated"):
            CustomModelWrapper(
                base_model=build_pymc_model(),
                idata=build_idata(),
                channels=CHANNELS,
            )

    def test_custom_model_wrapper_adstock_arg_warns(self):
        adstock = type("Adstock", (), {"l_max": 3})()
        with pytest.warns(DeprecationWarning, match="'adstock' argument is deprecated"):
            CustomModelWrapper(
                base_model=build_pymc_model(),
                idata=build_idata(),
                channels=CHANNELS,
                adstock=adstock,
            )

    def test_build_merged_model_warns(self):
        with pytest.warns(DeprecationWarning, match="BuildMergedModel is deprecated"):
            self._build_merged_model()

    def test_set_predictors_for_optimization_warns(self):
        merged = self._build_merged_model()

        with pytest.warns(
            DeprecationWarning, match="_set_predictors_for_optimization is deprecated"
        ):
            merged._set_predictors_for_optimization(num_periods=2)

    @staticmethod
    def _build_merged_model() -> BuildMergedModel:
        with pytest.warns(DeprecationWarning):
            wrapper = CustomModelWrapper(
                base_model=build_pymc_model(),
                idata=build_idata(),
                channels=CHANNELS,
            )
        with pytest.warns(DeprecationWarning):
            return BuildMergedModel(models=[wrapper], prefixes=["solo"])


class TestLegacyWrapperDispatch:
    """The wrapper dispatch must not depend on where the method sits in the MRO."""

    def test_concrete_method_ignores_protocol_stub(self):
        class StubOnly(OptimizerCompatibleModel):
            pass

        assert _concrete_method(StubOnly(), "optimization_model") is None
        assert _concrete_method(StubOnly(), "_set_predictors_for_optimization") is None

    def test_concrete_method_finds_inherited_implementation(self):
        class Base(OptimizerCompatibleModel):
            def optimization_model(self, num_periods: int) -> pm.Model:
                return build_pymc_model(num_periods)

        class Child(Base):
            """Inherits a concrete implementation rather than defining its own."""

        assert _concrete_method(Child(), "optimization_model") is not None

    def test_optimizer_accepts_wrapper_with_inherited_implementation(self):
        """A subclass that inherits ``optimization_model`` must still be unpacked."""

        class Base(OptimizerCompatibleModel):
            idata = build_idata()
            adstock_periods = 0

            def optimization_model(self, num_periods: int) -> pm.Model:
                return build_pymc_model(num_periods)

        class Child(Base):
            pass

        optimizer = BudgetOptimizer(model=Child(), num_periods=2)

        assert isinstance(optimizer.model, pm.Model)

    def test_stub_only_wrapper_raises_informative_error(self):
        class StubOnly(OptimizerCompatibleModel):
            idata = build_idata()

        with pytest.raises(ValueError, match="implements neither optimization_model"):
            BudgetOptimizer(model=StubOnly(), num_periods=2)


class TestChannelScalesValidation:
    """``channel_scales`` is checked up front rather than deep in the graph."""

    def _optimizer_kwargs(self, **overrides):
        kwargs = dict(model=build_pymc_model(), idata=build_idata(), num_periods=2)
        kwargs.update(overrides)
        return kwargs

    def test_scalar_is_accepted(self):
        optimizer = BudgetOptimizer(**self._optimizer_kwargs(channel_scales=2.0))

        assert optimizer.channel_scales == 2.0

    def test_matching_length_is_accepted(self):
        scales = np.array([1.0, 2.0])
        optimizer = BudgetOptimizer(**self._optimizer_kwargs(channel_scales=scales))

        np.testing.assert_array_equal(optimizer.channel_scales, scales)

    def test_wrong_length_raises(self):
        with pytest.raises(
            ValueError, match="channel_scales has length 3 but the model has 2 channels"
        ):
            BudgetOptimizer(
                **self._optimizer_kwargs(channel_scales=np.array([1.0, 2.0, 3.0]))
            )

    def test_too_many_dimensions_raises(self):
        with pytest.raises(
            ValueError, match="must be a scalar or a 1-D array over channels"
        ):
            BudgetOptimizer(**self._optimizer_kwargs(channel_scales=np.ones((2, 2))))
