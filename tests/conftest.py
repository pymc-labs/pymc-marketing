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
import pymc as pm
import pytest
from pymc.testing import mock_sample, mock_sample_setup_and_teardown


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run slow tests"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="only run slow tests"
    )
    parser.addoption(
        "--only-recovery",
        action="store_true",
        default=False,
        help="only run parameter-recovery tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "recovery: parameter-recovery / simulation study (real fits against a "
        "known ground truth). Runs nightly with --only-recovery; select with "
        "'-m recovery'. Implies slow, so plain pytest runs skip it",
    )
    config.addinivalue_line(
        "markers",
        "slow: too slow for the PR fast tier; deselected with '-m "
        '"not slow"\'. Run everything with --run-slow, or the per-PR slow '
        "tier with --only-slow ('-m \"slow and not recovery\"')",
    )
    config.addinivalue_line(
        "markers", "requires_polars: mark test as requiring polars library"
    )
    config.addinivalue_line(
        "markers", "requires_pyspark: mark test as requiring pyspark library"
    )


def pytest_collection_modifyitems(config, items):
    # Tier logic, with the marker-expression equivalents of each flag:
    #   plain pytest          == pytest -m "not slow"
    #   --run-slow            == no selection at all
    #   --only-slow           == pytest -m "slow and not recovery"
    #   --only-recovery       == pytest -m recovery
    # Recovery tests are a subset of slow tests
    for item in items:
        has_recovery = item.get_closest_marker("recovery") is not None
        has_slow = item.get_closest_marker("slow") is not None
        if has_recovery and not has_slow:
            item.add_marker(pytest.mark.slow)

    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not need to skip any tests
        return

    elif config.getoption("--only-recovery"):
        # --only-recovery given in cli: need to skip non-recovery tests
        skip_not_recovery = pytest.mark.skip(reason="Not a recovery test")
        for item in items:
            if item.get_closest_marker("recovery") is None:
                item.add_marker(skip_not_recovery)

    elif config.getoption("--only-slow"):
        # --only-slow given in cli: need to skip non-slow tests.
        # Recovery tests run separately (nightly), so skip them here too.
        skip_fast = pytest.mark.skip(reason="Fast test")
        skip_recovery = pytest.mark.skip(
            reason="Recovery test, use --only-recovery option to run"
        )
        for item in items:
            if item.get_closest_marker("recovery") is not None:
                item.add_marker(skip_recovery)
            elif item.get_closest_marker("slow") is None:
                item.add_marker(skip_fast)

    else:
        # Default: skip slow tests
        skip_slow = pytest.mark.skip(reason="Slow test, use --run-slow option to run")
        for item in items:
            if item.get_closest_marker("slow") is not None:
                item.add_marker(skip_slow)


mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)


@pytest.fixture
def mock_pymc_fit(mocker):
    """Replace `pm.fit` (the variational fit seam) with prior draws.

    Stands in for the `pm.fit` + `Approximation.sample` chain the wrapper
    performs: whatever `draws` and `chains` the test passes to ``fit`` are
    honored by drawing from the sampling model's own priors via
    `pymc.testing.mock_sample`, so no graph optimization or gradient
    compilation ever runs::

        def test_something(mock_pymc_fit):
            idata = model.fit(data, method="advi", draws=10)
    """

    def fake_fit(*args, **kwargs):
        fake_approx = mocker.Mock(name="pm.fit")

        def fake_sample(**sample_kwargs):
            return mock_sample(
                draws=sample_kwargs.get("draws", 10),
                chains=sample_kwargs.get("chains", 1),
                model=pm.modelcontext(None),
            )

        fake_approx.sample.side_effect = fake_sample
        return fake_approx

    return mocker.patch("pymc.fit", side_effect=fake_fit)


@pytest.fixture
def mock_pymc_find_map(mocker):
    """Replace `pm.find_MAP` (the MAP fit seam) with the initial point.

    Lets tests assert wiring — which model reaches the optimizer — without
    paying for the real minimization (`initial_point` shares find_MAP's
    return contract: value-var names -> arrays). The returned ``Mock``
    records ``call_args`` for such assertions.
    """
    return mocker.patch.object(
        pm,
        "find_MAP",
        side_effect=lambda *, model, **kwargs: model.initial_point(),
        name="find_MAP",
    )
