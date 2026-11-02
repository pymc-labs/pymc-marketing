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

"""The cache of compiled sweeps behind :meth:`SensitivityAnalysis.run_sweep`.

A :class:`SweepCache` keeps compiled sweep functions, keyed by the structure of
the question, and a call leases one entry for its duration so concurrent callers
never rebind each other's posterior draws.  The entry itself, its graph builders
and its key live in :mod:`pymc_marketing.mmm.sweep_graph`; this module holds no
graph code.
"""

from __future__ import annotations

from collections.abc import Callable
from weakref import WeakKeyDictionary

from pymc import Model

from pymc_marketing.mmm.leased_cache import LeasedLRUCache
from pymc_marketing.mmm.sweep_graph import (
    SweepCacheKey,
    SweepCloneError,
    SweepEntry,
)

DEFAULT_MAX_ENTRIES = 4
"""Number of compiled sweeps a cache keeps before evicting the least recent."""


class SweepLease:
    """One call's hold on a compiled sweep, taken out of the cache.

    A leased entry belongs to the call until :meth:`release`: nothing else can
    rebind the shared draws under it, so a concurrent call on the same key
    misses the cache and builds its own entry.  The entry is returned to the
    cache when the call is done, also when it raises, unless the call
    drops it with :meth:`discard`.

    Parameters
    ----------
    cache : SweepCache
        The cache it was taken from and is returned to.
    key : SweepCacheKey
        The identity it is cached under.
    entry : SweepEntry
        The compiled sweep.
    cached : bool
        Whether it was taken from the cache rather than built for this call.
    """

    def __init__(
        self,
        cache: SweepCache,
        key: SweepCacheKey,
        entry: SweepEntry,
        *,
        cached: bool,
    ) -> None:
        self._cache = cache
        self._key = key
        self._entry = entry
        self.cached = cached
        self._released = False

    @property
    def entry(self) -> SweepEntry:
        """The compiled sweep this lease holds."""
        return self._entry

    def release(self) -> None:
        """Return the entry to the cache, once.

        Releasing twice is a no-op, so both a failure path and the caller's
        ``finally`` may release.  If the cache already holds an entry for this
        key -- another call finished first -- one is kept and the LRU stays
        bounded.
        """
        if self._released:
            return
        self._released = True
        self._cache.put(self._key, self._entry)

    def discard(self) -> None:
        """Drop the entry instead of returning it to the cache.

        A cached entry that cannot be rebound to this call's draws was built on
        another posterior layout, so keeping it would make every later call
        fail the same way.  A later :meth:`release` is a no-op; the caller
        leases a fresh entry in place of this one.
        """
        self._released = True


class SweepCache(LeasedLRUCache[SweepCacheKey, SweepEntry]):
    """Bounded LRU cache of compiled sweeps, leased for the duration of a call.

    An MMM owns one cache, and every
    :class:`~pymc_marketing.mmm.sensitivity_analysis.SensitivityAnalysis`
    reached through ``mmm.sensitivity`` leases from it, so one cache serves one
    model.  The cache is bounded so a model cannot pin an unbounded number of
    compiled functions, and it is never serialized because compiled functions
    only make sense in-process.

    A call leases an entry for its duration: :meth:`lease` takes the entry out
    of the cache and the call returns it in a ``finally``.  Concurrent calls on
    one cache never share the draws an entry is rebound to -- one of them
    misses the cache and builds its own -- and no call waits on another.  A
    lock guards only the dictionary bookkeeping, never an evaluation.  An entry
    whose compiled graph reads a source input replaced since its build is
    rebuilt rather than leased.

    An entry is only reused for the source model it was built on, checked
    separately from the key: the key describes the sweep graph only.  Analyses
    on different models that share one cache and one key structure therefore
    rebuild each other's entry on every call, which gives correct results but
    no reuse.

    A model PyMC cannot clone is remembered, so later calls go straight to the
    per-call path instead of retrying the clone.  The memory is keyed by model
    identity and dropped when the model is, and it is never serialized.

    Parameters
    ----------
    max_entries : int, default 4
        Number of compiled sweeps kept before the least recently used is
        evicted.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        super().__init__(max_entries)
        self._unclonable: WeakKeyDictionary[Model, str] = WeakKeyDictionary()

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore ``max_entries`` with no entries and no clone failures."""
        super().__setstate__(state)
        self._unclonable = WeakKeyDictionary()

    def lease(
        self,
        key: SweepCacheKey,
        source_model: Model,
        factory: Callable[[], SweepEntry],
    ) -> SweepLease:
        """Take the entry cached under ``key``, building one on a miss.

        The entry is removed from the cache and stays out until the lease is
        released, so a second call for the same key does not find it and
        builds its own instead of sharing draws with the first.  An entry whose
        compiled graph reads a source buffer replaced since it was built is
        treated as a miss, because the clone holds the old buffer and the
        factory rebuilds from the live data.

        Parameters
        ----------
        key : SweepCacheKey
            Identity to look up.
        source_model : Model
            The model the call is about to use; an entry built on another
            model is never leased.
        factory : callable
            Builds the entry when the key is absent or stale.

        Returns
        -------
        SweepLease
            The entry and whether it came from the cache.  A caller that
            rebinds a cached entry to new draws needs to know the difference: a
            freshly built one is already conditioned on the draws it was asked
            for.

        Raises
        ------
        SweepCloneError
            If the factory cannot clone ``source_model``.  The model is
            remembered, so a later lease raises this without trying again.
        """
        entry = self._take(key)
        if entry is not None and not entry.is_stale(source_model):
            return SweepLease(self, key, entry, cached=True)
        self._raise_if_unclonable(source_model)
        try:
            built = factory()
        except SweepCloneError as err:
            self._remember_unclonable(source_model, err)
            raise
        return SweepLease(self, key, built, cached=False)

    def _raise_if_unclonable(self, source_model: Model) -> None:
        """Raise if a previous build could not clone ``source_model``."""
        with self._lock:
            message = self._unclonable.get(source_model)
        if message is not None:
            raise SweepCloneError(message)

    def _remember_unclonable(self, source_model: Model, error: SweepCloneError) -> None:
        """Record that ``source_model`` cannot be cloned, for this cache's lifetime."""
        with self._lock:
            self._unclonable[source_model] = str(error)
