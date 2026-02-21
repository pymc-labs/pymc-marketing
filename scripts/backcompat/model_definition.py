from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pymc_marketing.model_builder import ModelBuilder


@dataclass(frozen=True)
class ModelDefinition:
    """Container describing how to build and fit a model for backcompat checks."""

    name: str
    builder_cls: type[ModelBuilder]
    builder_fn: Callable[..., ModelBuilder]
    build_args_fn: Callable[[], dict[str, Any]]
    sampler_kwargs: dict[str, Any]
    fit_seed: int
    fit_args_fn: Callable[[], dict[str, Any]] | None = None
    fit_data_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
