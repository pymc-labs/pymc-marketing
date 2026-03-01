from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from ..model_definition import ModelDefinition


def _discover_models() -> dict[str, ModelDefinition]:
    """
    Auto-discover model definitions by scanning all Python files in this directory.

    Looks for modules that define a `get_model_definition()` function and automatically
    registers them. This eliminates the need for manual registration in __init__.py.

    Returns
    -------
    dict[str, ModelDefinition]
        Registry mapping model names to their definitions.

    Raises
    ------
    RuntimeError
        If a model module is missing the required `get_model_definition()` function.
    """
    registry = {}
    models_dir = Path(__file__).parent

    for model_file in models_dir.glob("*.py"):
        # Skip __init__.py and other special files
        if model_file.name.startswith("_"):
            continue

        module_name = model_file.stem
        try:
            # Import the module dynamically
            module = importlib.import_module(f".{module_name}", package=__package__)

            # Check if it has get_model_definition function
            if not hasattr(module, "get_model_definition"):
                raise RuntimeError(
                    f"Model module '{module_name}' must define a "
                    f"'get_model_definition()' function that returns a ModelDefinition. "
                    f"See scripts/backcompat/README.md for examples."
                )

            get_model_def = module.get_model_definition

            # Validate it's a callable
            if not callable(get_model_def):
                raise RuntimeError(
                    f"Model module '{module_name}' has 'get_model_definition' but it's not callable"
                )

            # Validate function signature (should take no arguments)
            sig = inspect.signature(get_model_def)
            if len(sig.parameters) > 0:
                raise RuntimeError(
                    f"Model module '{module_name}' function 'get_model_definition()' "
                    f"should take no arguments, but has parameters: {list(sig.parameters.keys())}"
                )

            # Call it to get the model definition
            model_def = get_model_def()

            # Validate it returns a ModelDefinition
            if not isinstance(model_def, ModelDefinition):
                raise RuntimeError(
                    f"Model module '{module_name}' function 'get_model_definition()' "
                    f"must return a ModelDefinition, got {type(model_def)}"
                )

            # Register using the model's name field (not the filename)
            registry[model_def.name] = model_def

        except ImportError as e:
            # Skip files that can't be imported (might be incomplete or have issues)
            print(f"Warning: Failed to import model module '{module_name}': {e}")
            continue

    return registry


# Auto-discover all models in this directory
MODEL_REGISTRY: dict[str, ModelDefinition] = _discover_models()


def get_model_definition(name: str) -> ModelDefinition:
    """
    Get a model definition by name.

    Parameters
    ----------
    name : str
        Model name (e.g., "basic_mmm", "beta_geo").

    Returns
    -------
    ModelDefinition
        The model definition.

    Raises
    ------
    ValueError
        If the model name is not found in the registry.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Valid options: {valid}") from exc
