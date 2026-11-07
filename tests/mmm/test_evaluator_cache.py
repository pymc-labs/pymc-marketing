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
"""Unit tests for the compiled-evaluator cache.

The cache only asks an entry whether it went stale, so these tests store
stand-ins.  How ``Incrementality`` uses the cache on a fitted MMM is tested in
``tests/mmm/test_incrementality.py``.
"""

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.evaluator_cache import (
    EvaluatorCache,
    EvaluatorCacheKey,
    constant_data_fingerprint,
)


class _FakeEvaluator:
    """Only what the evaluator cache asks of an entry: whether it went stale."""

    def __init__(self, stale: bool = False) -> None:
        self.stale = stale

    def is_stale(self) -> bool:
        """Whether the entry should be rebuilt instead of reused."""
        return self.stale


def _key(dates: tuple = (1, 2)) -> EvaluatorCacheKey:
    """A cache key for ``channel_contribution`` over ``dates``."""
    return EvaluatorCacheKey(
        response_vars=("channel_contribution",),
        intervention_target="channel_data",
        intervention_mode="replace",
        frozen_deterministics=(),
        dates=dates,
        data_fingerprint=None,
    )


def test_equal_keys_find_the_same_entry():
    """A key built again with the same fields finds the entry; other dates do not."""
    cache = EvaluatorCache()
    evaluator = _FakeEvaluator()
    cache.put(_key(), evaluator)

    assert cache.get(_key()) is evaluator
    assert cache.get(_key(dates=(1, 3))) is None


def test_the_lru_evicts_the_least_recent_and_get_refreshes_recency():
    """A size-4 cache drops its oldest key, and a ``get`` hit protects a key from the next eviction."""
    cache = EvaluatorCache(max_entries=4)
    keys = [object() for _ in range(5)]
    values = [object() for _ in range(5)]

    for key, value in zip(keys[:4], values[:4], strict=True):
        cache.put(key, value)

    cache.put(keys[4], values[4])

    assert len(cache) == 4
    assert cache.get(keys[0]) is None

    # A hit on the least recent key makes it the most recent, so the next put evicts the one after it.
    assert cache.get(keys[1]) is values[1]
    cache.put(keys[0], values[0])

    assert cache.get(keys[1]) is values[1]
    assert cache.get(keys[2]) is None


def test_put_replaces_an_existing_key_without_growing():
    cache = EvaluatorCache(max_entries=2)
    first, second = _FakeEvaluator(), _FakeEvaluator()

    cache.put(_key(), first)
    cache.put(_key(), second)

    assert len(cache) == 1
    assert cache.get(_key()) is second


def test_a_miss_builds_with_the_factory_and_the_release_caches_it():
    cache = EvaluatorCache()
    built = []

    def factory():
        built.append(_FakeEvaluator())
        return built[-1]

    lease = cache.lease(_key(), factory)

    assert not lease.cached
    assert lease.evaluator is built[0]
    assert len(cache) == 0

    lease.release()

    assert cache.get(_key()) is built[0]


def test_a_leased_entry_is_out_of_the_cache_until_released():
    """A second request on a leased key misses rather than sharing it."""
    cache = EvaluatorCache(max_entries=2)
    key = object()
    evaluator = _FakeEvaluator()
    cache.put(key, evaluator)

    lease = cache.lease(key, lambda: pytest.fail("a cached entry is a hit"))
    assert lease.cached
    assert lease.evaluator is evaluator
    assert len(cache) == 0

    concurrent = cache.lease(key, _FakeEvaluator)
    assert not concurrent.cached
    assert concurrent.evaluator is not evaluator

    lease.release()
    lease.release()  # releasing twice is a no-op
    concurrent.release()

    assert len(cache) == 1
    assert cache.get(key) is concurrent.evaluator


def test_a_stale_entry_is_rebuilt_instead_of_leased():
    """An evaluator that reads a replaced source input is dropped for a fresh build."""
    cache = EvaluatorCache()
    stale, fresh = _FakeEvaluator(stale=True), _FakeEvaluator()
    cache.put(_key(), stale)

    lease = cache.lease(_key(), lambda: fresh)

    assert not lease.cached
    assert lease.evaluator is fresh

    lease.release()

    assert len(cache) == 1
    assert cache.get(_key()) is fresh


@pytest.mark.parametrize(
    "duplicate",
    [
        # The test loads only what it pickled itself.
        pytest.param(lambda cache: pickle.loads(pickle.dumps(cache)), id="pickle"),  # noqa: S301
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_a_copy_starts_empty_and_keeps_its_bound(duplicate):
    """Compiled evaluators make sense only in-process, so a copy holds none.

    The copy still evicts at the original's ``max_entries`` and has a working
    lock of its own.
    """
    cache = EvaluatorCache(max_entries=2)
    cache.put(_key(), _FakeEvaluator())

    copied = duplicate(cache)

    assert len(copied) == 0
    assert len(cache) == 1

    for dates in [(1,), (2,), (3,)]:
        copied.put(_key(dates=dates), _FakeEvaluator())

    assert len(copied) == 2
    assert copied.get(_key(dates=(1,))) is None


def test_threaded_requests_keep_the_cache_bounded():
    """Concurrent leases, releases and puts never raise and never overfill the cache."""
    cache = EvaluatorCache(max_entries=3)
    keys = [_key(dates=(index,)) for index in range(6)]

    def request(index: int) -> None:
        lease = cache.lease(keys[index % len(keys)], _FakeEvaluator)
        cache.put(keys[(index + 1) % len(keys)], _FakeEvaluator())
        lease.release()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(request, range(400)))

    assert len(cache) == 3


def _constant_data(
    values=(1.0, 2.0), channel=("a", "b"), dim: str = "channel", dtype=float
) -> xr.DataTree:
    """A tree whose ``constant_data`` holds ``channel_data`` over ``dim``.

    The labels are stored as Python objects, as a model's string labels can
    be, so their array holds pointers rather than text.
    """
    constant_data = xr.Dataset(
        {"channel_data": ((dim,), np.asarray(values, dtype=dtype))},
        coords={dim: pd.Index(list(channel), dtype=object)},
    )
    return xr.DataTree.from_dict({"/constant_data": constant_data})


def test_constant_data_fingerprint_is_none_without_constant_data():
    idata = xr.DataTree.from_dict({"/posterior": xr.Dataset()})

    assert constant_data_fingerprint(idata) is None


def test_constant_data_fingerprint_is_equal_for_equal_content():
    """Two trees built separately with the same data share a fingerprint.

    Object labels are hashed by their text, not by the address of each
    string, so labels created twice still match.
    """
    first = _constant_data(channel=[f"channel {i}" for i in range(2)])
    second = _constant_data(channel=[f"channel {i}" for i in range(2)])
    first_labels = first["constant_data"].coords["channel"].values
    second_labels = second["constant_data"].coords["channel"].values
    # The same text in other string objects: hashing the buffer would differ.
    assert first_labels.dtype == object
    assert first_labels.tobytes() != second_labels.tobytes()

    assert constant_data_fingerprint(first) == constant_data_fingerprint(second)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(lambda: _constant_data(values=(1.0, 3.0)), id="value"),
        pytest.param(
            lambda: _constant_data(values=(2.0, 1.0), channel=("b", "a")),
            id="label-order",
        ),
        pytest.param(lambda: _constant_data(dim="geo"), id="dim-name"),
        pytest.param(lambda: _constant_data(dtype=np.float32), id="dtype"),
    ],
)
def test_constant_data_fingerprint_changes_with_the_data(changed):
    """A change of value, label order, dim name or dtype changes the fingerprint.

    The same data with its labels in another order is a different dataset to a
    compiled evaluator, which reads ``channel_data`` by position.
    """
    assert constant_data_fingerprint(changed()) != constant_data_fingerprint(
        _constant_data()
    )
