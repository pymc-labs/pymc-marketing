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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pymc_extras.statespace as pmss
import pytensor.tensor as pt
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pymc_extras.statespace.core import PyMCStateSpace
from pymc_extras.statespace.core.properties import Data as DataInfo
from pymc_extras.statespace.core.properties import Parameter, Shock, State
from scipy.stats import multivariate_normal

from pymc_marketing.experimental.terms import SCM, LatentStateSpace, Problem
from pymc_marketing.mmm import GeometricAdstock
from pymc_marketing.terms import Data, Equation, Observed
from pymc_marketing.terms._graph import walk

N = 30
A = np.array([[0.7, 0.2], [-0.1, 0.5]])
TREND = np.array([0.3, -0.2])
BETA = np.array([[0.5, -0.4], [1.5, 0.2]])  # (state, input), inputs ["x", "w"]
Q = np.array([[0.2, 0.05], [0.05, 0.1]])
X0 = np.array([0.1, -0.3])
P0 = np.diag([0.5, 0.4])
SIGMA = np.array([0.3, 0.2])

FIXED = {
    "x0": X0,
    "P0": P0,
    "trend_params": TREND[:, None],
    "ar_params": A[:, None, :],
    "state_cov": Q,
    "beta_exog": BETA,
}
VARMAX = {
    "order": (1, 0),
    "endog_names": ["a", "b"],
    "exog_state_names": ["x", "w"],
    "trend": "c",
    "measurement_error": False,
}


@pytest.fixture(scope="module")
def data() -> xr.Dataset:
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {
            "a": ("date", rng.normal(size=N)),
            "b": ("date", rng.normal(size=N)),
            "c": ("date", rng.normal(size=N)),
            "z": ("date", rng.normal(size=N)),
            # Stored in the other order than the model's inputs
            "inputs": (("date", "input"), rng.normal(size=(N, 2))),
        },
        coords={
            "date": pd.date_range("2024-01-01", periods=N, freq="W"),
            "input": ["w", "x"],
        },
    )


def latent(name="lat", model_kwargs=None, **priors) -> LatentStateSpace:
    return LatentStateSpace(
        name,
        pmss.BayesianVARMAX,
        model_kwargs={**VARMAX, **(model_kwargs or {})},
        priors={**FIXED, **priors},
        exog={"exogenous_data": Data("inputs")},
    )


def equations(lat, likelihood="Normal"):
    sigma = {"sigma" if likelihood == "Normal" else "s": SIGMA[0]}
    a = Equation(
        observed="a", mu=Data("z") + lat["a"], likelihood=Prior(likelihood, **sigma)
    )
    b = Equation(observed="b", mu=lat["b"], likelihood=Prior("Normal", sigma=SIGMA[1]))
    return a, b


def kalman_loglike(y, a0, P0, c, d, T, Z, H, Q) -> float:
    """Prediction-error decomposition in NumPy, with per-date c, d and Z.

    The state at date t is T x_{t-1} + c_t plus a shock, starting from a0 at the
    first date.
    """
    a, P, total = a0.copy(), P0.copy(), 0.0
    for t in range(len(y)):
        if t > 0:
            a, P = T @ a + c[t], T @ P @ T.T + Q
        mean, F = d[t] + Z[t] @ a, Z[t] @ P @ Z[t].T + H
        total += multivariate_normal(mean, F).logpdf(y[t])
        K = P @ Z[t].T @ np.linalg.inv(F)
        a, P = a + K @ (y[t] - mean), P - K @ Z[t] @ P
    return total


def varmax_loglike(ds: xr.Dataset, offset_b: np.ndarray | None = None) -> float:
    inputs = ds["inputs"].sel(input=["x", "w"]).values
    return kalman_loglike(
        np.column_stack([ds["a"].values, ds["b"].values]),
        X0,
        P0,
        TREND + inputs @ BETA.T,
        np.column_stack(
            [ds["z"].values, np.zeros(N) if offset_b is None else offset_b]
        ),
        A,
        np.broadcast_to(np.eye(2), (N, 2, 2)),
        np.diag(SIGMA**2),
        Q,
    )


def observation_logp(model: pm.Model, name: str = "lat", **point) -> float:
    return model.compile_logp(vars=[model[f"{name}_observations"]])(point)


@pytest.mark.parametrize(
    "filter_settings",
    [
        {"filter_type": "standard"},
        {"filter_type": "standard", "use_joseph_form": True},
        {"filter_type": "cholesky"},
    ],
    ids=["standard", "joseph", "cholesky"],
)
def test_marginal_likelihood_matches_numpy_filter(data, filter_settings):
    lat = latent(model_kwargs=filter_settings)
    model = SCM(*equations(lat)).build_model(data)
    assert not model.free_RVs
    # pymc-extras stabilises the observed covariance with a small jitter (cov_jitter)
    np.testing.assert_allclose(observation_logp(model), varmax_loglike(data), rtol=1e-6)


def test_filter_follows_the_model():
    lat = latent(model_kwargs={"filter_type": "cholesky", "cov_jitter": 1e-4})
    kalman_filter = lat.kalman_filter(SimpleNamespace(time_varying=()))
    assert type(kalman_filter).__name__ == "SquareRootFilter"
    assert kalman_filter.cov_jitter == 1e-4
    lat = latent(model_kwargs={"use_joseph_form": True})
    assert lat.kalman_filter(SimpleNamespace(time_varying=())).use_joseph_form


@pytest.mark.parametrize("filter_type", ["univariate", "convergent"])
def test_filters_that_cannot_take_per_date_intercepts_are_refused(filter_type):
    with pytest.raises(ValueError, match="'standard' and 'cholesky'"):
        latent(model_kwargs={"filter_type": filter_type})


def test_fit_records_the_final_filtered_state(data):
    lat = latent()
    exact = {"sigma": 1e-6}
    a = Equation(
        observed="a", mu=Data("z") + lat["a"], likelihood=Prior("Normal", **exact)
    )
    b = Equation(observed="b", mu=lat["b"], likelihood=Prior("Normal", **exact))
    model = SCM(a, b).build_model(data)
    last = data.isel(date=-1)
    # Observed exactly, the state at the last date is the last observation minus the offset
    np.testing.assert_allclose(
        pm.draw(model["lat_final_state"]), [last["a"] - last["z"], last["b"]], atol=1e-5
    )
    np.testing.assert_allclose(pm.draw(model["lat_final_state_cov"]), 0, atol=1e-5)


def test_observed_equations_are_marginalized(data):
    lat = latent()
    a = Observed("Normal", observed="a", mu=Data("z") + lat["a"], sigma=SIGMA[0])
    b = Observed("Normal", observed="b", mu=lat["b"], sigma=SIGMA[1])
    model = SCM(a, b).build_model(data)
    np.testing.assert_allclose(observation_logp(model), varmax_loglike(data), rtol=1e-6)


def test_equation_can_use_another_equation_of_the_same_latent(data):
    lat = latent()
    a, _ = equations(lat)
    b = Equation(
        observed="b", mu=lat["b"] + a, likelihood=Prior("Normal", sigma=SIGMA[1])
    )
    model = SCM(a, b).build_model(data)
    # a's value is its observations, a known offset for b
    np.testing.assert_allclose(
        observation_logp(model),
        varmax_loglike(data, offset_b=data["a"].values),
        rtol=1e-6,
    )


def test_prior_dims_are_aligned_by_name(data):
    lat = latent(
        ar_params=Prior(
            "Normal", dims=("lag_ar", "observed_state_aux", "observed_state")
        )
    )
    model = SCM(*equations(lat)).build_model(data)
    assert model.named_vars_to_dims["lat_ar_params"] == (
        "lat_lag_ar",
        "lat_observed_state_aux",
        "lat_observed_state",
    )
    # A laid out along the prior's own dims: [lag, source, target]
    point = {"lat_ar_params": A.T[None]}
    np.testing.assert_allclose(
        observation_logp(model, **point), varmax_loglike(data), rtol=1e-6
    )


def test_prior_function_returns_the_requested_dim_order(data):
    def ar_params(make):
        return make(
            "ar_raw",
            Prior("Normal", dims=("observed_state_aux", "lag_ar", "observed_state")),
            dims=("observed_state", "lag_ar", "observed_state_aux"),
        )

    lat = latent(ar_params=ar_params)
    model = SCM(*equations(lat)).build_model(data)
    # A laid out along the prior's own dims: [source, lag, target]
    point = {"lat_ar_raw": A.T[:, None, :]}
    np.testing.assert_allclose(
        observation_logp(model, **point), varmax_loglike(data), rtol=1e-6
    )


@pytest.mark.parametrize(
    "reorder",
    [
        lambda arr: arr.transpose("observed_state_aux", "lag_ar", "observed_state"),
        lambda arr: arr.sel(observed_state=["b", "a"]),
    ],
    ids=["transposed", "relabeled"],
)
def test_labeled_fixed_values_line_up_by_name(data, reorder):
    arr = xr.DataArray(
        A[:, None, :],
        dims=("observed_state", "lag_ar", "observed_state_aux"),
        coords={
            "observed_state": ["a", "b"],
            "lag_ar": [1],
            "observed_state_aux": ["a", "b"],
        },
    )
    model = SCM(*equations(latent(ar_params=reorder(arr)))).build_model(data)
    np.testing.assert_allclose(observation_logp(model), varmax_loglike(data), rtol=1e-6)


def test_covariance_parameters_refuse_a_prior(data):
    lat = latent(state_cov=Prior("HalfNormal", sigma=1))
    with pytest.raises(ValueError, match="positive semi-definite"):
        SCM(*equations(lat)).build_model(data)


def test_fixed_values_must_have_the_parameter_shape(data):
    # A diagonal given as a vector would broadcast to a non-symmetric covariance
    lat = latent(state_cov=np.array([0.2, 0.1]))
    with pytest.raises(ValueError):
        SCM(*equations(lat)).build_model(data)


def test_priors_use_prefixed_model_dims(data):
    def state_cov(make):
        sd = make(
            "shock_sd", Prior("HalfNormal", sigma=1, dims="shock"), dims=("shock",)
        )
        return pt.diag(sd**2)

    lat = latent(
        beta_exog=Prior(
            "Normal",
            mu=0,
            sigma=xr.DataArray([1.0, 2.0], dims=("observed_state",)),
            dims=("observed_state",),
        ),
        state_cov=state_cov,
    )
    model = SCM(*equations(lat)).build_model(data)
    dims = model.named_vars_to_dims
    assert dims["lat_beta_exog"] == ("lat_observed_state",)
    assert dims["lat_shock_sd"] == ("lat_shock",)
    assert dims["lat_state_cov"] == ("lat_shock", "lat_shock_aux")
    assert list(model.coords["lat_observed_state"]) == ["a", "b"]


def test_exogenous_labels_must_match_the_model(data):
    lat = LatentStateSpace(
        "lat",
        pmss.BayesianVARMAX,
        model_kwargs={**VARMAX, "exog_state_names": ["x", "v"]},
        priors=FIXED,
        exog={"exogenous_data": Data("inputs")},
    )
    with pytest.raises(ValueError, match="must match the model's inputs"):
        SCM(*equations(lat)).build_model(data)


class _TimeVaryingLoading(PyMCStateSpace):
    """y_t = level_t + loading_t x_t, both states random walks."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(k_endog=1, k_states=2, k_posdef=2, verbose=verbose)

    def set_parameters(self):
        return (
            Parameter("x0", shape=(2,), dims=("state",)),
            Parameter("P0", shape=(2, 2), dims=("state", "state_aux")),
            Parameter("sigma_state", shape=(2,), dims=("shock",)),
        )

    def set_states(self):
        return (
            State("level", observed=False),
            State("loading", observed=False),
            State("y", observed=True),
        )

    def set_shocks(self):
        return (Shock("level"), Shock("loading"))

    def set_data_info(self):
        return DataInfo("x", shape=(None,), dims=("time",), is_exogenous=True)

    def make_symbolic_graph(self) -> None:
        x0 = self.make_and_register_variable("x0", shape=(2,))
        P0 = self.make_and_register_variable("P0", shape=(2, 2))
        sigma = self.make_and_register_variable("sigma_state", shape=(2,))
        x = self.make_and_register_data("x", shape=(None,))
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = P0
        self.ssm["transition", :, :] = np.eye(2)
        self.ssm["selection", :, :] = np.eye(2)
        self.ssm["state_cov", :, :] = pt.diag(sigma**2)
        self.ssm["design"] = pt.stack([pt.ones_like(x), x], axis=-1)[:, None, :]
        self.ssm.declare_time_varying("design")


class _TimeVaryingShocks(PyMCStateSpace):
    """A random walk whose shock variance is data, one value per date."""

    def __init__(self, verbose: bool = False) -> None:
        super().__init__(k_endog=1, k_states=1, k_posdef=1, verbose=verbose)

    def set_parameters(self):
        return (
            Parameter("x0", shape=(1,), dims=("state",)),
            Parameter("P0", shape=(1, 1), dims=("state", "state_aux")),
        )

    def set_states(self):
        return (State("level", observed=False), State("y", observed=True))

    def set_shocks(self):
        return Shock("level")

    def set_data_info(self):
        return DataInfo("v", shape=(None,), dims=("time",), is_exogenous=True)

    def make_symbolic_graph(self) -> None:
        self.ssm["initial_state", :] = self.make_and_register_variable("x0", shape=(1,))
        self.ssm["initial_state_cov", :, :] = self.make_and_register_variable(
            "P0", shape=(1, 1)
        )
        v = self.make_and_register_data("v", shape=(None,))
        self.ssm["transition", :, :] = np.eye(1)
        self.ssm["selection", :, :] = np.eye(1)
        self.ssm["design", :, :] = np.eye(1)
        self.ssm["state_cov"] = v[:, None, None]
        self.ssm.declare_time_varying("state_cov")


def test_prior_predictive_shock_uses_the_previous_dates_variance(data):
    v = np.full(N, 0.01)
    v[1] = 100.0
    lat = LatentStateSpace(
        "lat",
        _TimeVaryingShocks,
        priors={"x0": np.zeros(1), "P0": np.eye(1) * 1e-6},
        exog={"v": Data("v")},
    )
    y = Equation(observed="c", mu=lat["y"], likelihood=Prior("Normal", sigma=0.1))
    _, model, _ = SCM(y)._training_model(
        data.assign(v=("date", v)), simulate_latents=True
    )
    draws = pm.draw(model["lat_latent"], draws=50, random_seed=7)
    # As in the filter, the step from date 0 to date 1 uses date 0's variance (0.01),
    # not date 1's (100)
    assert draws[:, 1, 0].var() < 1


def test_time_varying_design_matches_numpy_filter(data):
    sigma_state = np.array([0.3, 0.1])
    lat = LatentStateSpace(
        "lat",
        _TimeVaryingLoading,
        priors={"x0": X0, "P0": P0, "sigma_state": sigma_state},
        exog={"x": Data("z")},
    )
    y = Equation(observed="c", mu=lat["y"], likelihood=Prior("Normal", sigma=0.4))
    model = SCM(y).build_model(data)
    z = data["z"].values
    expected = kalman_loglike(
        data["c"].values[:, None],
        X0,
        P0,
        np.zeros((N, 2)),
        np.zeros((N, 1)),
        np.eye(2),
        np.stack([np.ones(N), z], axis=1)[:, None, :],
        np.array([[0.4**2]]),
        np.diag(sigma_state**2),
    )
    np.testing.assert_allclose(observation_logp(model), expected, rtol=1e-6)


def test_prior_predictive_draws_states_forward(data):
    lat = latent()
    draws = SCM(*equations(lat)).sample_prior_predictive(data, draws=20, random_seed=3)
    assert {"a", "b"} <= set(draws["prior_predictive"].data_vars)
    assert draws["prior"]["lat_latent"].dims[-2:] == ("date", "lat_state")

    # Without initial or shock variance the path is the deterministic recursion
    still = latent(P0=np.zeros((2, 2)), state_cov=np.zeros((2, 2)))
    _, model, _ = SCM(*equations(still))._training_model(data, simulate_latents=True)
    states = pm.draw(model["lat_latent"])
    inputs = data["inputs"].sel(input=["x", "w"]).values
    expected = [X0]
    for t in range(1, N):
        expected.append(A @ expected[-1] + TREND + BETA @ inputs[t])
    np.testing.assert_allclose(states, np.array(expected), rtol=1e-6, atol=1e-6)


def test_prior_predictive_ignores_the_observations(data):
    lat = latent()
    scm = SCM(*equations(lat))
    shifted = data.assign(a=data["a"] + 100.0, b=data["b"] - 100.0)
    first, second = (
        scm.sample_prior_predictive(ds, draws=10, random_seed=4)["prior_predictive"]
        for ds in (data, shifted)
    )
    xr.testing.assert_allclose(first["a"], second["a"])


def test_nested_observed_equations_are_marginalized(data):
    lat = latent()
    a, b = equations(lat)
    downstream = Equation(
        observed="c", mu=Data("z") + a, likelihood=Prior("Normal", sigma=1.0)
    )
    model = SCM(downstream, b).build_model(data)
    assert set(model.coords["lat_equation"]) == {"a", "b"}
    assert "c" in model.named_vars


def test_save_and_load_keep_the_marginalized_latent(data, tmp_path):
    lat = latent(
        trend_params=Prior("Normal", sigma=1, dims=("observed_state", "trend"))
    )
    scm = SCM(*equations(lat))
    scm.fit(data, draws=20, tune=20, chains=1, random_seed=5, progressbar=False)
    path = tmp_path / "model.zarr"
    scm.save(path)
    loaded = SCM.load(path)  # recomputes the joint log-density at saved draws
    (restored,) = {
        id(node): node
        for node in walk(loaded.equations)
        if isinstance(node, LatentStateSpace)
    }.values()
    assert restored.model is pmss.BayesianVARMAX
    assert restored.model_kwargs == VARMAX
    assert "lat_observations" in loaded.model.named_vars


@pytest.fixture(scope="module")
def fitted_with_latent(data):
    lat = latent(
        trend_params=Prior("Normal", sigma=1, dims=("observed_state", "trend"))
    )
    a, b = equations(lat)
    scm = SCM(a, b)
    scm.fit(data, draws=10, tune=10, chains=1, random_seed=1, progressbar=False)
    return scm, lat, next(term for term in a.mu.terms if isinstance(term, Data))


def test_optimizer_runs_on_terms_that_do_not_touch_the_latent(data, fitted_with_latent):
    scm, _, z = fitted_with_latent
    dates = data.get_index("date")[-4:]
    problem = Problem(
        model=scm,
        terms={"z": z},
        dates=dates,
        objective=lambda evaluate, u: evaluate(u)["z"].sum(),
        bounds={"z": (0.0, 1.0)},
    )
    plan = xr.Dataset({"z": ("date", np.full(4, 0.5))}, coords={"date": dates})
    assert "z" in problem.evaluate(plan)


def future(data: xr.Dataset, n: int = 4) -> xr.Dataset:
    """Inputs for the n dates after the data."""
    rng = np.random.default_rng(2)
    dates = pd.date_range(data.get_index("date")[-1], periods=n + 1, freq="W")[1:]
    return xr.Dataset(
        {
            "z": ("date", rng.normal(size=n)),
            "inputs": (("date", "input"), rng.normal(size=(n, 2))),
        },
        coords={"date": dates, "input": ["w", "x"]},
    )


NOISE = Prior("Normal", sigma=Prior("HalfNormal", sigma=1))


def fit_without_noise(data, *equations_):
    """Fit, then zero the equations' noise and the recorded state's uncertainty.

    With no shocks either, a forecast is then exactly its expected path.
    """
    scm = SCM(*equations_)
    scm.fit(data, draws=5, tune=5, chains=1, random_seed=1, progressbar=False)
    posterior = scm.idata["posterior"]
    for name in posterior.data_vars:
        if name.endswith(("_sigma", "_final_state_cov")):
            posterior[name] = posterior[name] * 0
    return scm


@pytest.fixture(scope="module")
def fitted_without_shocks(data):
    """Fitted with no initial-state or shock variance, so the states are a recursion."""
    lat = latent(
        P0=np.zeros((2, 2)),
        state_cov=np.zeros((2, 2)),
        trend_params=Prior("Normal", sigma=1, dims=("observed_state", "trend")),
    )
    return fit_without_noise(
        data,
        Equation(observed="a", mu=Data("z") + lat["a"], likelihood=NOISE),
        Equation(observed="b", mu=lat["b"], likelihood=NOISE),
    )


def recursion(start, trend, inputs):
    """States x_t = A x_{t-1} + trend + BETA u_t, for each draw, from ``start``."""
    states, x = [], start
    for u in inputs.sel(input=["x", "w"]).values:
        x = x @ A.T + trend + u @ BETA.T
        states.append(x)
    return np.stack(states, axis=1)


def test_forecast_starts_from_the_recorded_state(data, fitted_without_shocks):
    scm = fitted_without_shocks
    posterior = scm.idata["posterior"].isel(chain=0)
    new = future(data)
    draws = scm.sample_posterior_predictive(new).isel(chain=0)
    expected = recursion(
        posterior["lat_final_state"].values,
        posterior["lat_trend_params"].values[..., 0],
        new["inputs"],
    )
    np.testing.assert_allclose(
        draws["a"].values, new["z"].values + expected[..., 0], atol=1e-8
    )
    np.testing.assert_allclose(draws["b"].values, expected[..., 1], atol=1e-8)


def test_forecast_carries_the_recorded_state_uncertainty(data):
    # Noisy observations and small shocks make the state at the last fitted date
    # uncertain; zeroing the equations' noise after fitting leaves that uncertainty
    lat = latent(
        state_cov=0.01 * np.eye(2),
        trend_params=Prior("Normal", sigma=1, dims=("observed_state", "trend")),
    )
    scm = SCM(
        Equation(observed="a", mu=Data("z") + lat["a"], likelihood=NOISE),
        Equation(observed="b", mu=lat["b"], likelihood=NOISE),
    )
    scm.fit(data, draws=1000, tune=200, chains=1, random_seed=1, progressbar=False)
    posterior = scm.idata["posterior"]
    for name in ("a_sigma", "b_sigma"):
        posterior[name] = posterior[name] * 0
    posterior = posterior.isel(chain=0)
    new = future(data)
    draws = scm.sample_posterior_predictive(new, random_seed=2).isel(chain=0, date=0)
    mean, cov = (
        posterior["lat_final_state"].values,
        posterior["lat_final_state_cov"].values,
    )
    trend = posterior["lat_trend_params"].values[..., 0]
    inputs = new["inputs"].sel(input=["x", "w"]).values[0]
    forecast = np.column_stack(
        [draws["a"].values - new["z"].values[0], draws["b"].values]
    )
    residual = forecast - (mean @ A.T + trend + inputs @ BETA.T)
    # One step ahead: the recorded uncertainty moved forward, plus one shock
    expected = (A @ cov @ A.T).mean(axis=0) + 0.01 * np.eye(2)
    np.testing.assert_allclose(np.cov(residual.T), expected, atol=0.01)


def test_forecast_continues_a_time_trend(data):
    lat = latent(
        model_kwargs={"trend": "ct"},
        P0=np.zeros((2, 2)),
        state_cov=np.zeros((2, 2)),
        trend_params=Prior("Normal", sigma=0.1, dims=("observed_state", "trend")),
    )
    scm = fit_without_noise(
        data,
        Equation(observed="a", mu=Data("z") + lat["a"], likelihood=NOISE),
        Equation(observed="b", mu=lat["b"], likelihood=NOISE),
    )
    new = future(data)
    draws = scm.sample_posterior_predictive(new).isel(chain=0)
    posterior = scm.idata["posterior"].isel(chain=0)
    trend = posterior["lat_trend_params"].values  # (draw, state, [constant, slope])
    # The trend's time keeps counting from the first fitted date, as when fitting
    x, expected = posterior["lat_final_state"].values, []
    for k, u in enumerate(new["inputs"].sel(input=["x", "w"]).values):
        time = N + k + 1  # pymc-extras' trend_offset=1, one date ahead like the inputs
        x = x @ A.T + trend[..., 0] + trend[..., 1] * time + u @ BETA.T
        expected.append(x)
    np.testing.assert_allclose(
        draws["b"].values, np.stack(expected, axis=1)[..., 1], atol=1e-8
    )


def test_forecast_without_history_starts_from_the_initial_state(
    data, fitted_without_shocks
):
    scm = fitted_without_shocks
    trend = scm.idata["posterior"].isel(chain=0)["lat_trend_params"].values[..., 0]
    new = future(data)
    draws = scm.sample_posterior_predictive(new, include_last_observations=False)
    # The first date is the initial state; nothing is carried over from training
    first = np.broadcast_to(X0, trend.shape)
    expected = np.concatenate(
        [
            first[:, None],
            recursion(first, trend, new["inputs"].isel(date=slice(1, None))),
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        draws["b"].isel(chain=0).values, expected[..., 1], atol=1e-8
    )


def test_forecast_returns_the_equations_on_the_new_dates(data, fitted_with_latent):
    scm, _, _ = fitted_with_latent
    new = future(data)
    draws = scm.sample_posterior_predictive(new, random_seed=2)
    assert set(draws.data_vars) == {"a", "b"}
    assert draws.get_index("date").equals(new.get_index("date"))
    assert np.isfinite(draws["a"]).all() and np.isfinite(draws["b"]).all()


def test_forecast_after_save_and_load(data, fitted_with_latent, tmp_path):
    scm, _, _ = fitted_with_latent
    scm.save(tmp_path / "scm.zarr")
    loaded = SCM.load(tmp_path / "scm.zarr")
    new = future(data)
    xr.testing.assert_allclose(
        loaded.sample_posterior_predictive(new, random_seed=2),
        scm.sample_posterior_predictive(new, random_seed=2),
    )


def test_conditioning_on_an_observing_equation_does_not_inform_the_latent(
    data, fitted_with_latent
):
    scm, _, _ = fitted_with_latent
    draws = [
        scm.sample_posterior_predictive(
            future(data).assign(a=("date", np.full(4, value))),
            condition_on=["a"],
            random_seed=2,
        )
        for value in (0.0, 5.0)
    ]
    np.testing.assert_array_equal(draws[1]["a"].isel(chain=0, draw=0), 5.0)
    # a is held, as for any other equation: b is forecast from the fitted data alone
    xr.testing.assert_equal(draws[0]["b"], draws[1]["b"])


def test_both_filters_record_the_same_final_state(data):
    models = [
        SCM(*equations(latent(model_kwargs={"filter_type": kind}))).build_model(data)
        for kind in ("standard", "cholesky")
    ]
    for name in ("lat_final_state", "lat_final_state_cov"):
        standard, cholesky = (pm.draw(model[name]) for model in models)
        assert np.abs(standard).max() > 1e-3
        np.testing.assert_allclose(cholesky, standard, rtol=1e-6, atol=1e-10)


def test_forecast_with_a_time_varying_loading(data):
    lat = LatentStateSpace(
        "lat",
        _TimeVaryingLoading,
        priors={"x0": X0, "P0": np.zeros((2, 2)), "sigma_state": np.zeros(2)},
        exog={"x": Data("z")},
    )
    scm = fit_without_noise(data, Equation(observed="c", mu=lat["y"], likelihood=NOISE))
    new = future(data)
    draws = scm.sample_posterior_predictive(new).isel(chain=0)
    level, loading = scm.idata["posterior"]["lat_final_state"].isel(chain=0).values.T
    # Constant states, observed through each forecast date's own loading
    np.testing.assert_allclose(
        draws["c"].values,
        level[:, None] + loading[:, None] * new["z"].values,
        atol=1e-8,
    )


def test_forecast_through_an_adstock_on_the_latent_input(data):
    adstock = GeometricAdstock(
        l_max=3, priors={"alpha": Prior("Beta", alpha=2, beta=2)}
    )
    lat = LatentStateSpace(
        "lat",
        pmss.BayesianVARMAX,
        model_kwargs=VARMAX,
        priors={
            **FIXED,
            "P0": np.zeros((2, 2)),
            "state_cov": np.zeros((2, 2)),
            "trend_params": Prior("Normal", sigma=1, dims=("observed_state", "trend")),
        },
        exog={"exogenous_data": Data("inputs") >> adstock},
    )
    scm = fit_without_noise(
        data,
        Equation(observed="a", mu=Data("z") + lat["a"], likelihood=NOISE),
        Equation(observed="b", mu=lat["b"], likelihood=NOISE),
    )
    new = future(data)
    draws = scm.sample_posterior_predictive(new).isel(chain=0)

    posterior = scm.idata["posterior"].isel(chain=0)
    alpha = next(
        posterior[name] for name in posterior.data_vars if name.endswith("alpha")
    )
    alpha = (
        alpha.broadcast_like(data["inputs"].isel(date=0))
        .sel(input=["x", "w"])
        .transpose("draw", "input")
    )
    # The adstocked inputs on the forecast dates carry over the last fitted ones
    inputs = (
        xr.concat([data["inputs"], new["inputs"]], dim="date")
        .sel(input=["x", "w"])
        .values
    )
    weights = alpha.values[..., None] ** np.arange(3)  # (draw, input, lag)
    weights /= weights.sum(axis=-1, keepdims=True)
    adstocked = sum(
        weights[:, None, :, lag] * inputs[None, N - lag : N + 4 - lag]
        for lag in range(3)
    )  # (draw, date, input)
    states, x = [], posterior["lat_final_state"].values
    trend = posterior["lat_trend_params"].values[..., 0]
    for t in range(4):
        x = x @ A.T + trend + np.einsum("di,si->ds", adstocked[:, t], BETA)
        states.append(x)
    expected = np.stack(states, axis=1)
    np.testing.assert_allclose(draws["b"].values, expected[..., 1], atol=1e-8)


def test_optimizing_through_a_latent_is_refused(data, fitted_with_latent):
    scm, lat, _ = fitted_with_latent
    with pytest.raises(NotImplementedError, match="latent state-space terms"):
        Problem(
            model=scm,
            terms={"state": lat["a"]},
            dates=data.get_index("date")[-4:],
            objective=lambda evaluate, u: evaluate(u)["state"].sum(),
            bounds={"inputs": (0.0, 1.0)},
        )


def test_parameters_and_data_must_match_the_model():
    with pytest.raises(ValueError, match="missing"):
        LatentStateSpace(
            "lat",
            pmss.BayesianVARMAX,
            model_kwargs=VARMAX,
            priors={},
            exog={"exogenous_data": Data("inputs")},
        )
    with pytest.raises(ValueError, match="data name"):
        LatentStateSpace("lat", pmss.BayesianVARMAX, model_kwargs=VARMAX, priors=FIXED)


def test_unknown_component_lists_the_observed_states():
    with pytest.raises(KeyError, match=r"\['a', 'b'\]"):
        latent()["c"]


def test_latent_can_only_enter_the_mean(data):
    lat = latent()
    a = Equation(
        observed="a",
        mu=lat["a"],
        parameters={"sigma": lat["b"]},
        likelihood=Prior("Normal"),
    )
    with pytest.raises(ValueError, match="must be Normal"):
        SCM(a, equations(lat)[1]).build_model(data)


def test_marginalization_needs_normal_likelihood(data):
    lat = latent()
    scm = SCM(*equations(lat, likelihood="Logistic"))
    with pytest.raises(ValueError, match="must be Normal"):
        scm.build_model(data)
    # The prior predictive refuses what training refuses
    with pytest.raises(ValueError, match="must be Normal"):
        scm.sample_prior_predictive(data, draws=5)


def test_marginalization_refuses_transformed_likelihood(data):
    lat = latent()
    a = Equation(
        observed="a",
        mu=lat["a"],
        likelihood=Prior("Normal", sigma=SIGMA[0], transform="exp"),
    )
    with pytest.raises(ValueError, match="transformed or noncentered"):
        SCM(a, equations(lat)[1]).build_model(data)


def test_equation_cannot_observe_two_latents(data):
    first, second = latent("one"), latent("two")
    both = Equation(
        observed="a", mu=first["a"] + second["a"], likelihood=Prior("Normal", sigma=1.0)
    )
    with pytest.raises(ValueError, match="more than one marginalized latent"):
        SCM(both).build_model(data)
