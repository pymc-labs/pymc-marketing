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

"""The compiled-evaluator cache a fitted MMM keeps for its incrementality questions.

A fitted :class:`~pymc_marketing.mmm.mmm.MMM` owns one :class:`EvaluatorCache`.
Every :class:`~pymc_marketing.mmm.incrementality.Incrementality` object it
creates leases compiled
:class:`~pymc_marketing.mmm.counterfactual.CounterfactualEvaluator` objects from
it, so repeated questions over the same graph and data compile once.  The
model creates and clears the cache itself; a user does not call it directly.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from pymc_marketing.mmm.counterfactual import CounterfactualEvaluator


class EvaluatorCacheKey(NamedTuple):
    """Identity of a compiled counterfactual evaluator.

    Structural fields pick the compiled graph; ``dates`` and ``data_fingerprint`` pick the data it was built and
    checked against, so two wrappers over different data never share an entry.  A response is keyed by its name:
    the evaluator resolves every response by name against its intervened clone, so the name is what it compiles.
    """

    response_vars: tuple[str, ...]
    intervention_target: str
    intervention_mode: str
    frozen_deterministics: tuple[str, ...]
    dates: tuple
    data_fingerprint: tuple | None


def _digest(values: Any) -> bytes:
    """Return a 16-byte BLAKE2b digest of an array's content."""
    array = np.asarray(values)
    # A label array's buffer holds pointers rather than content, so hash its text; numeric arrays hash directly.
    payload = (
        repr(array.tolist()).encode()
        if array.dtype.hasobject
        else np.ascontiguousarray(array).tobytes()
    )
    return hashlib.blake2b(payload, digest_size=16).digest()


def constant_data_fingerprint(idata: xr.DataTree) -> tuple | None:
    """Return a content fingerprint of a DataTree's fit-time ``constant_data``.

    Each ``constant_data`` variable is compared by name, dims, shape, dtype and a digest of its values, and each
    coordinate by name, dims and a digest of its labels.  Since ``channel_data`` is included, wrappers differing
    only in it, or only in the order of its labels, compile separately, while the summary factory and
    ``MMM.incrementality`` share an entry because they wrap one fit-time dataset.

    Parameters
    ----------
    idata : xr.DataTree
        Fitted inference data whose ``constant_data`` group is fingerprinted when present.

    Returns
    -------
    tuple or None
        One entry per data variable and per coordinate.  ``None`` when ``idata`` has no ``constant_data`` group.
    """
    constant_data = getattr(idata, "constant_data", None)
    if constant_data is None:
        return None
    variables = tuple(
        (
            name,
            tuple(map(str, constant_data[name].dims)),
            constant_data[name].shape,
            np.asarray(constant_data[name].values).dtype.str,
            _digest(constant_data[name].values),
        )
        for name in sorted(map(str, constant_data.data_vars))
    )
    coords = tuple(
        (
            name,
            tuple(map(str, constant_data.coords[name].dims)),
            _digest(constant_data.coords[name].values),
        )
        for name in sorted(map(str, constant_data.coords))
    )
    return variables + coords


class EvaluatorLease:
    """One request's hold on a compiled evaluator, taken out of the cache.

    A leased evaluator belongs to the request until :meth:`release`: nothing
    else can rebind the shared draws under it, so a concurrent request misses
    the cache and compiles or leases a copy of its own.  The evaluator is
    returned to the cache when the request is done, also when it raises.

    Parameters
    ----------
    cache : EvaluatorCache
        The cache it was taken from and is returned to.
    key : EvaluatorCacheKey
        The identity it is cached under.
    evaluator : CounterfactualEvaluator
        The compiled evaluator.
    cached : bool
        Whether it was taken from the cache rather than built for this request.
    """

    def __init__(
        self,
        cache: EvaluatorCache,
        key: EvaluatorCacheKey,
        evaluator: CounterfactualEvaluator,
        *,
        cached: bool,
    ) -> None:
        self._cache = cache
        self._key = key
        self._evaluator = evaluator
        self.cached = cached
        self._released = False

    @property
    def evaluator(self) -> CounterfactualEvaluator:
        """The compiled evaluator this lease holds."""
        return self._evaluator

    def release(self) -> None:
        """Return the evaluator to the cache, once.

        Releasing twice is a no-op, so both a failure path and the caller's
        ``finally`` may release.  If the cache already holds an evaluator for
        this key -- another request finished first -- one is kept and the LRU
        stays bounded.
        """
        if self._released:
            return
        self._released = True
        self._cache.put(self._key, self._evaluator)


class EvaluatorCache:
    """Bounded LRU cache of compiled counterfactual evaluators.

    A fitted MMM keeps one so its ``Incrementality`` objects reuse compiled evaluators, keyed by graph shape and
    dataset, so repeated questions pay one compile.
    The cache is bounded so a model cannot pin an unbounded number of compiled functions, and it is never serialized
    because compiled functions only make sense in-process.
    A request leases an evaluator for its duration: :meth:`lease` takes the entry out of the cache and the request
    returns it in a ``finally``.  Concurrent requests on one model never share the shared draws an evaluator
    is rebound to -- one of them misses the cache and compiles or leases its own -- and no request waits on another.
    A lock guards only the dictionary bookkeeping, never an evaluation.
    An entry whose evaluator reads a source input replaced after its build is rebuilt rather than leased.

    Parameters
    ----------
    max_entries : int, default 4
        Number of evaluators kept before the least recently used is evicted.
    """

    def __init__(self, max_entries: int = 4) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[EvaluatorCacheKey, CounterfactualEvaluator] = (
            OrderedDict()
        )
        # Guards only the dictionary bookkeeping, never a request: a release must not
        # fail because another thread evicted its key between two dict operations.
        self._lock = threading.Lock()

    def get(self, key: EvaluatorCacheKey) -> CounterfactualEvaluator | None:
        """Return the evaluator cached under ``key``, marking it most recent.

        Parameters
        ----------
        key : EvaluatorCacheKey
            Identity to look up.

        Returns
        -------
        CounterfactualEvaluator or None
            The cached evaluator, which stays in the cache, or ``None`` when
            ``key`` is absent.  A request takes an evaluator out with
            :meth:`lease` instead.
        """
        with self._lock:
            evaluator = self._entries.get(key)
            if evaluator is not None:
                self._entries.move_to_end(key)
            return evaluator

    def put(self, key: EvaluatorCacheKey, evaluator: CounterfactualEvaluator) -> None:
        """Store ``evaluator`` under ``key``, evicting the least recent entry.

        An evaluator already cached under ``key`` is replaced, so the cache
        never holds more than ``max_entries`` evaluators.

        Parameters
        ----------
        key : EvaluatorCacheKey
            Identity to store it under.
        evaluator : CounterfactualEvaluator
            The compiled evaluator.
        """
        with self._lock:
            self._entries[key] = evaluator
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def lease(
        self,
        key: EvaluatorCacheKey,
        factory: Callable[[], CounterfactualEvaluator],
    ) -> EvaluatorLease:
        """Take the evaluator cached under ``key``, building one on a miss.

        The entry is removed from the cache and stays out until the lease is
        released, so a second request for the same key does not find it and
        builds its own instead of sharing draws with the first.  A cached
        evaluator that reads a source input replaced since it was built is
        treated as a miss: the compiled graphs hold the old buffer, so the
        factory rebuilds it.

        Parameters
        ----------
        key : EvaluatorCacheKey
            Identity to look up.
        factory : callable
            Builds the evaluator when the key is absent or stale.

        Returns
        -------
        EvaluatorLease
            The evaluator and whether it came from the cache.  A caller that
            rebinds a cached evaluator to new draws needs to know the
            difference: a freshly built one is already conditioned on the draws
            it was asked for.
        """
        with self._lock:
            evaluator = self._entries.pop(key, None)
        if evaluator is not None and not evaluator.is_stale():
            return EvaluatorLease(self, key, evaluator, cached=True)
        return EvaluatorLease(self, key, factory(), cached=False)

    def __len__(self) -> int:
        """Return the number of cached evaluators."""
        return len(self._entries)

    def __getstate__(self) -> dict[str, int]:
        """Skip the compiled evaluators when the owning model is pickled."""
        return {"_max_entries": self._max_entries}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore `max_entries` with an empty cache."""
        self._max_entries = state["_max_entries"]
        self._entries = OrderedDict()
        self._lock = threading.Lock()
