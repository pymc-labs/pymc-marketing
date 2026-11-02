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
"""Unit tests for the sweep cache and the storage both caches share.

How the caches behave on a fitted MMM is tested in
``tests/mmm/test_sensitivity_analysis.py`` and
``tests/mmm/test_incrementality.py``.
"""

import copy
import pickle

import pytest

from pymc_marketing.mmm.evaluator_cache import EvaluatorCache
from pymc_marketing.mmm.sweep_cache import SweepCache, SweepCloneError

CACHE_TYPES = pytest.mark.parametrize(
    "cache_type", [EvaluatorCache, SweepCache], ids=["evaluator", "sweep"]
)

COPY_METHODS = pytest.mark.parametrize(
    "duplicate",
    [
        # The test loads only what it pickled itself.
        pytest.param(lambda cache: pickle.loads(pickle.dumps(cache)), id="pickle"),  # noqa: S301
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)


class _FakeSweepEntry:
    """Only what the sweep cache asks of an entry: whether it is stale for a model."""

    def __init__(self, source_model: object) -> None:
        self.source_model = source_model

    def is_stale(self, source_model: object) -> bool:
        """Whether the entry was built on another model."""
        return source_model is not self.source_model


class _FakeModel:
    """Only what the cache asks of a model: weak-referenceable identity."""


def test_a_leased_sweep_is_out_of_the_cache_until_released():
    """A second lease on a held key misses and builds, without threads."""
    cache = SweepCache()
    key = object()
    model = _FakeModel()
    built: list[_FakeSweepEntry] = []

    def factory() -> _FakeSweepEntry:
        built.append(_FakeSweepEntry(model))
        return built[-1]

    lease = cache.lease(key, model, factory)

    assert not lease.cached
    assert lease.entry is built[0]
    assert len(cache) == 0

    concurrent = cache.lease(key, model, factory)

    assert not concurrent.cached
    assert concurrent.entry is not lease.entry
    assert len(built) == 2

    lease.release()
    lease.release()  # releasing twice is a no-op
    concurrent.release()

    assert len(cache) == 1
    assert cache.get(key) is concurrent.entry


def test_a_sweep_entry_built_on_another_model_is_not_leased():
    """Only an entry built on the call's own source model is reused."""
    cache = SweepCache()
    key = object()
    built_on, other = _FakeModel(), _FakeModel()
    cached = _FakeSweepEntry(built_on)
    cache.put(key, cached)

    same_model = cache.lease(key, built_on, lambda: pytest.fail("expected a hit"))
    assert same_model.cached
    assert same_model.entry is cached
    same_model.release()

    fresh = _FakeSweepEntry(other)
    lease = cache.lease(key, other, lambda: fresh)

    assert not lease.cached
    assert lease.entry is fresh


@CACHE_TYPES
def test_a_negative_max_entries_is_refused(cache_type):
    with pytest.raises(ValueError, match="max_entries"):
        cache_type(max_entries=-1)


class _RecordingLock:
    """A context manager that counts how often it was entered."""

    def __init__(self) -> None:
        self.entered = 0

    def __enter__(self) -> "_RecordingLock":
        self.entered += 1
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


@CACHE_TYPES
def test_len_is_taken_under_the_lock(cache_type):
    cache = cache_type()
    lock = _RecordingLock()
    cache._lock = lock

    assert len(cache) == 0
    assert lock.entered == 1


@CACHE_TYPES
@COPY_METHODS
def test_a_copy_starts_empty_and_keeps_its_bound(cache_type, duplicate):
    """Entries are in-process only, but the copy still evicts at the same bound."""
    cache = cache_type(max_entries=2)
    cache.put(object(), object())

    copied = duplicate(cache)

    assert len(copied) == 0
    assert len(cache) == 1

    for _ in range(3):
        copied.put(object(), object())

    assert len(copied) == 2


def _failing_factory(attempts: list[int]):
    """A sweep factory that records the attempt and reports an unclonable model."""

    def factory() -> object:
        attempts.append(1)
        raise SweepCloneError("Cannot convert models with non-default initial_values")

    return factory


def test_an_unclonable_model_is_not_retried():
    """The cache remembers a model it could not clone and raises without retrying."""
    cache = SweepCache()
    model = _FakeModel()
    attempts: list[int] = []

    for _ in range(2):
        with pytest.raises(SweepCloneError):
            cache.lease(object(), model, _failing_factory(attempts))

    assert len(attempts) == 1


@COPY_METHODS
def test_the_unclonable_memo_does_not_survive_a_copy(duplicate):
    """A pickle or deepcopy of the cache starts with no remembered failures."""
    cache = SweepCache()
    model = _FakeModel()
    with pytest.raises(SweepCloneError):
        cache.lease(object(), model, _failing_factory([]))

    copied = duplicate(cache)
    attempts: list[int] = []

    with pytest.raises(SweepCloneError):
        copied.lease(object(), model, _failing_factory(attempts))

    assert len(attempts) == 1
