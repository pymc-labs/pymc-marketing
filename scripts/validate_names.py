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

Three checks run over ``pymc_marketing/``:

1. A string literal in code must not equal a value of a names module that owns
   the file, and must not end with one of its suffixes. A names module owns a
   file when the file is in its directory (see ``SCOPES``) or imports from it;
   the shared names own every file. Docstrings, ``__all__`` and ``Literal[...]``
   options are exempt. So is an ``idata.attrs`` or serialization key passed to
   ``getattr`` / ``hasattr`` / ``setattr`` as an attribute name: such a key lives
   in a dictionary, so the string names an attribute of a Python object there.
   Groups, variables, dims and columns get no such exemption, because xarray and
   pandas expose them as attributes. Any other exception needs an entry in
   ``ALLOWED`` for that file.
2. The MMM dims in ``DEFERRED_DIMS`` may stay literal where a typo raises
   (``dims=...``, subscripts, reductions). In a position where a typo fails
   silently (``in``, ``==``, ``.get``, ``hasattr``, set literals, ...) they
   follow check 1.
3. A names constant must not be used inside a function without annotations,
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
SCOPES = {
    "mmm": "mmm",
    "data": "mmm",
    "experimental": "mmm",
    "clv": "clv",
    "customer_choice": "customer_choice",
    "bass": "bass",
}
SHARED = "shared"

# MMM dims that keep the literal where a typo raises (follow-up of #3190).
DEFERRED_DIMS = frozenset({"date", "channel", "control", "cv", "sweep"})

_KEYWORD = "__init__ keyword, spelled where the keyword arguments are built"
_CONFIG_KEY = "model_config key, which stays a literal"
_LABEL = "plot label"
_DEFAULT = "default of a name the caller chooses"
_COLUMN = "DataFrame column, config key or option with the same string, not the dim"
_PARAMETER = "name of a distribution parameter"
_SCRATCH = "scratch variable that never reaches the model or idata"
_COMPONENT = "prefix of the component and name of its sampled curve"
_OWN_MODEL = "name of a model that keeps its own names"

# File (relative to ``pymc_marketing/``) -> {value: why the literal may stay there}.
ALLOWED: dict[str, dict[str, str]] = {
    "bass/plotting.py": {"T": _LABEL, "imitators": _LABEL, "innovators": _LABEL},
    "clv/models/beta_geo.py": {
        "dropout_coefficient": _CONFIG_KEY,
        "purchase_coefficient": _CONFIG_KEY,
    },
    "clv/models/modified_beta_geo.py": {
        "dropout_coefficient": _CONFIG_KEY,
        "purchase_coefficient": _CONFIG_KEY,
    },
    "clv/models/pareto_nbd.py": {
        "dropout_coefficient": _CONFIG_KEY,
        "purchase_coefficient": _CONFIG_KEY,
    },
    "clv/models/shifted_beta_geo.py": {
        "dropout_coefficient": _CONFIG_KEY,
        "dropout": "name of the sBG likelihood, a different variable from Var.DROPOUT",
    },
    "clv/plotting.py": {"T": _LABEL},
    "clv/utils.py": {"customer_id": _DEFAULT},
    "customer_choice/_choice_data_mixin.py": {
        "covariates": _KEYWORD,
        "depvar": _KEYWORD,
        "utility_equations": _KEYWORD,
        "model_config": _KEYWORD,
        "sampler_config": _KEYWORD,
    },
    "customer_choice/bayesian_blp.py": {"likelihood": "attrs key of the BLP model"},
    "customer_choice/maxdiff.py": {
        "U": _OWN_MODEL,
        "X": _OWN_MODEL,
        "model_config": _KEYWORD,
        "sampler_config": _KEYWORD,
    },
    "customer_choice/mixed_logit.py": {"likelihood": _CONFIG_KEY},
    "customer_choice/mnl_logit.py": {
        "X": "name of the instance attribute that holds the data",
        "covariates": _KEYWORD,
        "depvar": _KEYWORD,
        "utility_equations": _KEYWORD,
        "likelihood": _CONFIG_KEY,
    },
    "customer_choice/mv_its.py": {
        "likelihood": _CONFIG_KEY,
        "y": _OWN_MODEL,
        "model_config": _KEYWORD,
        "sampler_config": _KEYWORD,
    },
    "customer_choice/nested_logit.py": {
        "covariates": _KEYWORD,
        "depvar": _KEYWORD,
        "utility_equations": _KEYWORD,
        "likelihood": _CONFIG_KEY,
    },
    "customer_choice/synthetic_data.py": {"X": "key of the synthetic data dictionary"},
    "experimental/terms/_graph.py": {"mu": _PARAMETER},
    "mlflow.py": {"model_type": "name of an MLflow parameter"},
    "mmm/budget_optimizer.py": {
        "cost_per_unit": "label and tensor name of the cost table",
        "date": _COLUMN,
        "mu_effects": "removed keyword that is still recognised",
    },
    "mmm/builders/factories.py": {"model_config": "YAML key"},
    "mmm/components/adstock.py": {"adstock": _COMPONENT},
    "mmm/components/saturation.py": {"saturation": _COMPONENT},
    "mmm/counterfactual.py": {"time_index": "label of the batched input tensor"},
    "mmm/events.py": {"saturation": _COMPONENT},
    "mmm/fourier.py": {"fourier_mode": _DEFAULT},
    "mmm/incrementality.py": {
        "adstock": "name of the component passed to a check",
        "saturation": "name of the component passed to a check",
        "allocation": "estimand option",
        "link": "key of the reported assumptions",
        "time_varying_media": "key of the reported assumptions",
    },
    "mmm/lift_test.py": {"channel": _COLUMN, "date": _COLUMN, "mu": _PARAMETER},
    "mmm/media_transformation.py": {
        "adstock": "field of the serialized media transformation",
        "saturation": "field of the serialized media transformation",
    },
    "mmm/mmm.py": {
        "adstock": _KEYWORD,
        "saturation": _KEYWORD,
        "cost_per_unit": _KEYWORD,
        "link": _KEYWORD,
        "model_config": _KEYWORD,
        "sampler_config": _KEYWORD,
        "time_varying_intercept": _KEYWORD,
        "time_varying_media": _KEYWORD,
        "channel": _COLUMN,
        "date": _COLUMN,
        "y": _DEFAULT,
    },
    "mmm/plot.py": {"date": _COLUMN, "y_original_scale": _LABEL},
    "mmm/plot_interactive.py": {"channel_scale": _SCRATCH, "date": _COLUMN},
    "mmm/plotting/decomposition.py": {"y": "matplotlib keyword a caller may not set"},
    "mmm/plotting/diagnostics.py": {"y": _SCRATCH},
    "mmm/plotting/transformations.py": {"y": _SCRATCH},
    "mmm/summary/cv.py": {"cv": _COLUMN, "date": _COLUMN},
    "mmm/summary/factory.py": {"channel": _COLUMN, "control": _COLUMN, "y": _SCRATCH},
    "mmm/summary/sensitivity.py": {"sweep": _COLUMN},
    "mmm/synthetic_data/endogenous_spend.py": {"y": "column of the synthetic data"},
    "mmm/utils.py": {"adstock": "name of an MMM attribute in a check"},
    "model_builder.py": {"model_config": _KEYWORD, "sampler_config": _KEYWORD},
    "pie/model.py": {"model_config": _KEYWORD},
}

ATTRIBUTE_NAME_FUNCTIONS = {"getattr", "hasattr", "setattr", "delattr"}
# Names classes whose values are dictionary keys and never attributes. Only their
# values may be typed as the attribute name in ``getattr`` and the like.
DICTIONARY_KEY_CLASSES = {"Attr", "Key"}
SILENT_METHODS = {"get", "pop", "startswith", "endswith", "add", "discard"}
SILENT_KEYWORDS = {"errors", "missing_dims"}


def _string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


NamesModule = tuple[dict[str, list[str]], list[str], set[str]]


def read_names_module(path: Path) -> NamesModule:
    """Return the values, the suffixes and the public identifiers of a names module.

    Parameters
    ----------
    path : Path
        A module of ``pymc_marketing/names``.

    Returns
    -------
    NamesModule
        ``{value: ["Class.ATTRIBUTE", ...]}`` for every class attribute, the values
        of the module-level ``*_SUFFIX`` constants, and the public names the module
        defines.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    values: dict[str, list[str]] = {}
    suffixes: list[str] = []
    identifiers: set[str] = set()

    def assignments(body: list[ast.stmt]) -> Iterator[tuple[str, ast.expr | None]]:
        for statement in body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                yield statement.target.id, statement.value
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        yield target.id, statement.value

    for statement in tree.body:
        if isinstance(statement, ast.ClassDef):
            identifiers.add(statement.name)
            for attribute, expression in assignments(statement.body):
                value = _string(expression)
                if value is not None:
                    values.setdefault(value, []).append(f"{statement.name}.{attribute}")
        elif isinstance(statement, ast.FunctionDef):
            identifiers.add(statement.name)
    for name, expression in assignments(tree.body):
        identifiers.add(name)
        value = _string(expression)
        if value is not None and name.endswith("_SUFFIX"):
            suffixes.append(value)
    return values, suffixes, identifiers


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {
        id(child): node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }


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
    return exempt


def _attribute_names(tree: ast.AST) -> set[int]:
    """Collect the ids of the nodes passed as the attribute name to ``getattr`` etc."""
    return {
        id(node.args[1])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ATTRIBUTE_NAME_FUNCTIONS
        and len(node.args) > 1
    }


def _is_silent(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Return whether a typo in the string ``node`` would fail without raising."""
    parent = parents.get(id(node))
    if isinstance(parent, (ast.Tuple, ast.List)):
        node, parent = parent, parents.get(id(parent))
    if isinstance(parent, ast.Set):
        return True
    if isinstance(parent, ast.Compare):
        return any(
            isinstance(operator, (ast.In, ast.NotIn, ast.Eq, ast.NotEq))
            for operator in parent.ops
        )
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and len(parent.args) > 1
        and parent.args[1] is node
    ):
        if parent.func.id == "hasattr":
            return True
        if parent.func.id == "getattr":
            return len(parent.args) > 2  # a default hides the typo
    if isinstance(parent, ast.Call) and node in parent.args:
        if (
            isinstance(parent.func, ast.Attribute)
            and parent.func.attr in SILENT_METHODS
            and parent.args[0] is node
        ):
            return True
        return any(
            keyword.arg in SILENT_KEYWORDS and _string(keyword.value) == "ignore"
            for keyword in parent.keywords
        )
    return False


def _literals(tree: ast.AST) -> Iterator[tuple[ast.Constant, bool]]:
    """Yield the string nodes of ``tree`` and whether each is an f-string part."""
    fragments = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node, id(node) in fragments


def _names_imports(tree: ast.AST, identifiers: set[str]) -> tuple[set[str], set[str]]:
    """Return the local names and the names modules that ``tree`` imports."""
    local_names: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == "pymc_marketing":
            local_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "names"
            )
        elif node.module == "pymc_marketing.names":
            for alias in node.names:
                if alias.name in SCOPES.values():
                    modules.add(alias.name)
                    local_names.add(alias.asname or alias.name)
                elif alias.name in identifiers:
                    local_names.add(alias.asname or alias.name)
        elif node.module.startswith("pymc_marketing.names."):
            modules.add(node.module.rsplit(".", 1)[-1])
            local_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in identifiers
            )
    return local_names, modules


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

    def visit(
        node: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef | None
    ) -> Iterator[tuple[int, str]]:
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


def read_names() -> dict[str, NamesModule]:
    """Read the shared names module and the names module of every model package."""
    found = sorted({path.stem for path in NAMES_DIR.glob("*.py")} - {"__init__"})
    return {
        SHARED: read_names_module(NAMES_DIR / "__init__.py"),
        **{name: read_names_module(NAMES_DIR / f"{name}.py") for name in found},
    }


def check_source(
    relative: str, source: str, names: dict[str, NamesModule]
) -> tuple[list[str], set[str]]:
    """Check the source of one file.

    Parameters
    ----------
    relative : str
        Path of the file relative to ``pymc_marketing/``, with forward slashes.
    source : str
        Source code of the file.
    names : dict[str, NamesModule]
        Result of :func:`read_names`.

    Returns
    -------
    tuple[list[str], set[str]]
        The violations, and the ``ALLOWED`` values of this file that were used.
    """
    violations: list[str] = []
    used: set[str] = set()
    location = f"pymc_marketing/{relative}"
    tree = ast.parse(source, filename=location)
    identifiers = set().union(*(module[2] for module in names.values()))
    imported, imported_modules = _names_imports(tree, identifiers)
    directory = SCOPES.get(relative.split("/")[0]) if "/" in relative else None
    owners = [SHARED, *sorted({directory, *imported_modules} - {None})]
    values: dict[str, list[tuple[str, str]]] = {}
    for owner in owners:
        for value, constants in names[owner][0].items():
            values.setdefault(value, []).extend((owner, name) for name in constants)
    suffixes = [suffix for owner in owners for suffix in names[owner][1]]
    exempt = _exempt_nodes(tree)
    attribute_names = _attribute_names(tree)
    parents = _parents(tree)
    allowed = ALLOWED.get(relative, {})

    for node, is_fragment in _literals(tree):
        if id(node) in exempt:
            continue
        text = node.value
        if not is_fragment and text in values:
            if id(node) in attribute_names and all(
                constant.split(".")[0] in DICTIONARY_KEY_CLASSES
                for _, constant in values[text]
            ):
                continue
            deferred = "mmm" in owners and text in DEFERRED_DIMS
            if deferred and not _is_silent(node, parents):
                continue
            if text in allowed:
                used.add(text)
                continue
            position = " in a position where a typo fails silently" if deferred else ""
            owner, constant = values[text][0]
            violations.append(
                f"{location}:{node.lineno}: literal {text!r}{position} is "
                f"{constant} in the {owner} names"
            )
        elif any(text.endswith(suffix) for suffix in suffixes):
            violations.append(
                f"{location}:{node.lineno}: {text!r} spells a names suffix by hand"
            )

    for line, function in _unannotated_uses(tree, imported):
        violations.append(
            f"{location}:{line}: names constant inside {function}(), which has no "
            "annotations, so mypy does not check it"
        )
    return violations, used


def main() -> int:
    """Run the checks and print the violations."""
    violations: list[str] = []
    names = read_names()
    for name in sorted((set(names) - {SHARED}) ^ set(SCOPES.values())):
        violations.append(
            f"scripts/validate_names.py: names module {name!r} and SCOPES do not match"
        )
    used_allowances: set[tuple[str, str]] = set()

    for path in sorted(PACKAGE.rglob("*.py")):
        if NAMES_DIR in path.parents:
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        found, used = check_source(relative, path.read_text(), names)
        violations.extend(found)
        used_allowances.update((relative, text) for text in used)

    every_allowance = {
        (file, text) for file, entries in ALLOWED.items() for text in entries
    }
    for file, text in sorted(every_allowance - used_allowances):
        violations.append(
            f"scripts/validate_names.py: ALLOWED[{file!r}][{text!r}] matches nothing"
        )

    if violations:
        print("\n".join(violations))
        print(
            f"\n{len(violations)} problem(s). Use the constant from pymc_marketing/names, "
            "annotate the function, or add an ALLOWED entry for the file with a reason."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
