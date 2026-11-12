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
"""Validate that the names in ``pymc_marketing/names`` are not typed by hand again.

Two checks run over ``pymc_marketing/``:

1. A string literal in code must not equal a value that the names module of the
   same package defines, and must not end with one of its suffixes. Docstrings,
   ``__all__``, ``Literal[...]`` options, attribute names passed to ``getattr`` /
   ``hasattr`` / ``setattr`` and parameter defaults are exempt. Any other
   exception needs an entry in ``ALLOWED`` with a reason.
2. A names constant must not be used inside a function without annotations,
   because ``mypy`` does not check the body of such a function.

The script reads the names modules with ``ast`` and imports nothing from
``pymc_marketing``.

Usage:
    python scripts/validate_names.py

Exit codes:
    0 - No violations
    1 - Violations, or an ``ALLOWED`` entry that no longer matches anything
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "pymc_marketing"
NAMES_DIR = PACKAGE / "names"

# Directories under ``pymc_marketing/`` and the names module that owns their names.
# Files outside these directories are checked against the shared names only.
SCOPES = {
    "mmm": "mmm",
    "data": "mmm",
    "experimental": "mmm",
    "clv": "clv",
    "customer_choice": "customer_choice",
    "bass": "bass",
}
SHARED = "shared"

_DEFERRED_DIM = (
    "MMM dim: the literal stays where a typo raises and where it is a DataFrame "
    "column (follow-up of #3190)"
)
_INIT_KEYWORD = (
    "also an __init__ keyword, spelled where the keyword arguments are built"
)
_CONFIG_KEY = "also a model_config key, which stays a literal"
_PLOT_LABEL = "plot label"
_OWN_MODEL = "name of a model that keeps its own names"

# names module -> {value: why the literal may stay in the files of that scope}.
ALLOWED: dict[str, dict[str, str]] = {
    SHARED: {
        "model_config": _INIT_KEYWORD,
        "sampler_config": _INIT_KEYWORD,
        "model_type": "name of an MLflow parameter",
    },
    "mmm": {
        "date": _DEFERRED_DIM,
        "channel": _DEFERRED_DIM,
        "control": _DEFERRED_DIM,
        "cv": _DEFERRED_DIM,
        "sweep": _DEFERRED_DIM,
        "model_config": "__init__ keyword and YAML key",
        "sampler_config": _INIT_KEYWORD,
        "link": "__init__ keyword and key of the Incrementality assumptions",
        "cost_per_unit": "__init__ keyword and label in the budget optimizer",
        "mu_effects": "removed BudgetOptimizer keyword that is still recognised",
        "adstock": "component prefix, __init__ keyword and media transformation field",
        "saturation": "component prefix, __init__ keyword and media transformation field",
        "time_varying_intercept": _INIT_KEYWORD,
        "time_varying_media": "__init__ keyword and key of the Incrementality assumptions",
        "mu": "name of a distribution parameter",
        "y": "scratch variable in plots and synthetic data",
        "allocation": "estimand option of Incrementality",
        "channel_scale": "scratch DataFrame column in plot_interactive",
        "time_index": "label of the batched input in CounterfactualEvaluator",
        "y_original_scale": _PLOT_LABEL,
    },
    "clv": {
        "dropout_coefficient": _CONFIG_KEY,
        "purchase_coefficient": _CONFIG_KEY,
        "dropout": "name of the sBG likelihood, a different variable from Var.DROPOUT",
        "T": _PLOT_LABEL,
    },
    "customer_choice": {
        "model_config": _INIT_KEYWORD,
        "sampler_config": _INIT_KEYWORD,
        "utility_equations": _INIT_KEYWORD,
        "depvar": _INIT_KEYWORD,
        "covariates": _INIT_KEYWORD,
        "likelihood": "model_config key, and an attrs key of the BLP model",
        "y": _OWN_MODEL + " (MVITS)",
        "U": _OWN_MODEL + " (MaxDiff)",
        "X": "MaxDiff data container, attribute name and synthetic data key",
    },
    "bass": {
        "T": _PLOT_LABEL,
        "innovators": _PLOT_LABEL,
        "imitators": _PLOT_LABEL,
    },
}

ATTRIBUTE_NAME_FUNCTIONS = {"getattr", "hasattr", "setattr", "delattr"}


def _string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def read_names_module(path: Path) -> tuple[dict[str, str], list[str], set[str]]:
    """Return the values, the suffixes and the public identifiers of a names module.

    Parameters
    ----------
    path : Path
        A module of ``pymc_marketing/names``.

    Returns
    -------
    tuple[dict[str, str], list[str], set[str]]
        ``{value: "Class.ATTRIBUTE"}`` for every class attribute, the values of the
        module-level ``*_SUFFIX`` constants, and the public names the module defines.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    values: dict[str, str] = {}
    suffixes: list[str] = []
    identifiers: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.ClassDef):
            identifiers.add(statement.name)
            for item in statement.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    value = _string(item.value) if item.value is not None else None
                    if value is not None:
                        values.setdefault(value, f"{statement.name}.{item.target.id}")
        elif isinstance(statement, ast.FunctionDef):
            identifiers.add(statement.name)
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            identifiers.add(statement.target.id)
            value = _string(statement.value) if statement.value is not None else None
            if value is not None and statement.target.id.endswith("_SUFFIX"):
                suffixes.append(value)
        elif isinstance(statement, ast.Assign):
            identifiers.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
    return values, suffixes, identifiers


def scope_of(path: Path) -> str:
    """Return the names module that owns the names of ``path``."""
    parts = path.relative_to(PACKAGE).parts
    return SCOPES.get(parts[0], SHARED) if len(parts) > 1 else SHARED


def _exempt_nodes(tree: ast.AST) -> set[int]:
    """Collect the ids of the string nodes that the rule exempts."""
    exempt: set[int] = set()

    def exempt_subtree(node: ast.AST | None) -> None:
        if node is not None:
            exempt.update(id(child) for child in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and _string(node.value) is not None:
            exempt.add(id(node.value))  # docstring or attribute docstring
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exempt_subtree(node.value)
        elif isinstance(node, ast.Subscript) and (
            (isinstance(node.value, ast.Name) and node.value.id == "Literal")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "Literal")
        ):
            exempt_subtree(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ATTRIBUTE_NAME_FUNCTIONS
            and len(node.args) > 1
        ):
            exempt.add(id(node.args[1]))
        elif isinstance(node, ast.arguments):
            for default in (*node.defaults, *node.kw_defaults):
                exempt_subtree(default)
    return exempt


def _literals(tree: ast.AST) -> Iterator[tuple[ast.Constant, bool]]:
    """Yield the string nodes of ``tree`` and whether each is an f-string part."""
    fragments = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    for node in ast.walk(tree):
        if _string(node) is not None:
            yield node, id(node) in fragments  # type: ignore[misc]


def _names_imports(tree: ast.AST, identifiers: set[str]) -> set[str]:
    """Return the local names that ``tree`` imports from the names package."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "pymc_marketing.names" or node.module.startswith(
                "pymc_marketing.names."
            ):
                imported.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in identifiers
                )
    return imported


def _is_annotated(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    arguments = function.args
    every_argument = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        *([arguments.vararg] if arguments.vararg else []),
        *([arguments.kwarg] if arguments.kwarg else []),
    ]
    return function.returns is not None or any(
        argument.annotation is not None for argument in every_argument
    )


def _unannotated_uses(tree: ast.AST, imported: set[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(line, function name)`` for names constants in unannotated functions."""

    def visit(node: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef | None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node
        elif (
            isinstance(node, ast.Name)
            and node.id in imported
            and function is not None
            and not _is_annotated(function)
        ):
            yield node.lineno, function.name
        for child in ast.iter_child_nodes(node):
            yield from visit(child, function)

    yield from visit(tree, None)


def main() -> int:
    """Run both checks and print the violations."""
    modules = {
        SHARED: read_names_module(NAMES_DIR / "__init__.py"),
        **{
            name: read_names_module(NAMES_DIR / f"{name}.py")
            for name in sorted(set(SCOPES.values()))
        },
    }
    identifiers = set().union(*(module[2] for module in modules.values()))
    shared_values = modules[SHARED][0]

    violations: list[str] = []
    used_allowances: set[tuple[str, str]] = set()

    for path in sorted(PACKAGE.rglob("*.py")):
        if NAMES_DIR in path.parents:
            continue
        relative = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(), filename=str(path))
        scope = scope_of(path)
        values = {**shared_values, **modules[scope][0]}
        suffixes = modules[scope][1]
        exempt = _exempt_nodes(tree)

        for node, is_fragment in _literals(tree):
            if id(node) in exempt:
                continue
            text = node.value
            if not is_fragment and text in values:
                if text in ALLOWED.get(scope, {}):
                    used_allowances.add((scope, text))
                    continue
                owner = scope if text in modules[scope][0] else SHARED
                violations.append(
                    f"{relative}:{node.lineno}: literal {text!r} is "
                    f"{values[text]} in the {owner} names module"
                )
            elif any(text.endswith(suffix) for suffix in suffixes):
                violations.append(
                    f"{relative}:{node.lineno}: {text!r} spells a names suffix by hand"
                )

        imported = _names_imports(tree, identifiers)
        for line, function in _unannotated_uses(tree, imported):
            violations.append(
                f"{relative}:{line}: names constant inside {function}(), which has no "
                "annotations, so mypy does not check it"
            )

    every_allowance = {
        (scope, text) for scope, entries in ALLOWED.items() for text in entries
    }
    for scope, text in sorted(every_allowance - used_allowances):
        violations.append(
            f"scripts/validate_names.py: ALLOWED[{scope!r}][{text!r}] matches nothing"
        )

    if violations:
        print("\n".join(violations))
        print(
            f"\n{len(violations)} problem(s). Use the constant from pymc_marketing/names, "
            "annotate the function, or add an ALLOWED entry with a reason."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
