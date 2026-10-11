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
import pytest
from pymc.testing import mock_sample_setup_and_teardown


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
        "recovery: parameter-recovery / simulation study, runs only with "
        "--only-recovery or --run-slow",
    )
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "requires_polars: mark test as requiring polars library"
    )
    config.addinivalue_line(
        "markers", "requires_pyspark: mark test as requiring pyspark library"
    )


def pytest_collection_modifyitems(config, items):
    # Recovery tests are a subset of slow tests
    for item in items:
        if "recovery" in item.keywords and "slow" not in item.keywords:
            item.add_marker(pytest.mark.slow)

    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not need to skip any tests
        return

    elif config.getoption("--only-recovery"):
        # --only-recovery given in cli: need to skip non-recovery tests
        skip_not_recovery = pytest.mark.skip(reason="Not a recovery test")
        for item in items:
            if "recovery" not in item.keywords:
                item.add_marker(skip_not_recovery)

    elif config.getoption("--only-slow"):
        # --only-slow given in cli: need to skip non-slow tests.
        # Recovery tests run separately (nightly), so skip them here too.
        skip_fast = pytest.mark.skip(reason="Fast test")
        skip_recovery = pytest.mark.skip(
            reason="Recovery test, use --only-recovery option to run"
        )
        for item in items:
            if "recovery" in item.keywords:
                item.add_marker(skip_recovery)
            elif "slow" not in item.keywords:
                item.add_marker(skip_fast)

    else:
        # Default: skip slow tests
        skip_slow = pytest.mark.skip(reason="Slow test, use --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)
