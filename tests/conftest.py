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
import dataclasses
from contextlib import contextmanager

import pytest
from pymc.testing import mock_sample_setup_and_teardown
from pymc_extras.prior import Scaled

from pymc_marketing.serialization import serialization


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="also run slow tests"
    )
    parser.addoption(
        "--only-slow", action="store_true", default=False, help="only run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "requires_polars: mark test as requiring polars library"
    )
    config.addinivalue_line(
        "markers", "requires_pyspark: mark test as requiring pyspark library"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not need to skip any tests
        return

    elif config.getoption("--only-slow"):
        # --only-slow given in cli: need to skip non-slow tests
        skip_fast = pytest.mark.skip(reason="Fast test")
        for item in items:
            if "slow" not in item.keywords:
                item.add_marker(skip_fast)

    else:
        # Default: skip slow tests
        skip_slow = pytest.mark.skip(reason="Slow test, use --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


mock_pymc_sample = pytest.fixture(scope="module")(mock_sample_setup_and_teardown)


@pytest.fixture
def writing_scaled_as_in_1_1_0(monkeypatch):
    """Return a context manager under which saves write ``Scaled`` as 1.1.0 did.

    1.1.0 wrote ``{"__type__": "pymc_extras.prior.Scaled", "dist": ..., "factor": ...}``.
    Saving under it gives a file with both that shape and the model id hashed from it.
    """
    key = serialization.key_for(Scaled)

    def serialize_1_1_0(obj: Scaled) -> dict:
        return {"__type__": key, "dist": obj.dist.to_dict(), "factor": obj.factor}

    @contextmanager
    def writing():
        entry = serialization._registry[key]
        with monkeypatch.context() as m:
            m.setitem(
                serialization._registry,
                key,
                dataclasses.replace(entry, serializer=serialize_1_1_0),
            )
            yield

    return writing
