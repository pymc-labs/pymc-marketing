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

1. A string literal in code must not equal a value of a names module that owns
   the file, and must not end with one of its suffixes. A names module owns a
   file when the file is in its directory (see ``SCOPES``) or imports from it;
   the shared names own every file. Docstrings, ``__all__`` and ``Literal[...]``
   options are exempt. So is an ``idata.attrs`` or serialization key passed to
   ``getattr`` / ``hasattr`` / ``setattr`` as an attribute name: such a key lives
   in a dictionary, so the string names an attribute of a Python object there.
   Groups, variables, dims and columns get no such exemption, because xarray and
   pandas expose them as attributes. Any other exception needs an entry in
   ``ALLOWED`` for that file, with a reason and the number of such literals in
   the file, so a new literal with the same value still fails.
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
SCOPES = {
    "mmm": "mmm",
    "data": "mmm",
    "experimental": "mmm",
    "clv": "clv",
    "customer_choice": "customer_choice",
    "bass": "bass",
}
SHARED = "shared"

_KEYWORD = "__init__ keyword, spelled where the keyword arguments are built"
_CONFIG_KEY = "model_config key, which stays a literal"
_LABEL = "plot label"
_DEFAULT = "default of a name the caller chooses"
_COLUMN = "DataFrame column, config key or option with the same string, not the dim"
_PARAMETER = "name of a distribution parameter"
_SCRATCH = "scratch variable that never reaches the model or idata"
_COMPONENT = "prefix of the component and name of its sampled curve"
_OWN_MODEL = "name of a model that keeps its own names"

# File (relative to ``pymc_marketing/``) -> {value: (why the literal may stay there,
# how many times it appears in the file)}.
ALLOWED: dict[str, dict[str, tuple[str, int]]] = {
    "bass/plotting.py": {
        "T": (_LABEL, 3),
        "imitators": (_LABEL, 1),
        "innovators": (_LABEL, 1),
    },
    "clv/models/beta_geo.py": {
        "dropout_coefficient": (_CONFIG_KEY, 7),
        "purchase_coefficient": (_CONFIG_KEY, 3),
    },
    "clv/models/modified_beta_geo.py": {
        "dropout_coefficient": (_CONFIG_KEY, 6),
        "purchase_coefficient": (_CONFIG_KEY, 2),
    },
    "clv/models/pareto_nbd.py": {
        "dropout_coefficient": (_CONFIG_KEY, 3),
        "purchase_coefficient": (_CONFIG_KEY, 3),
    },
    "clv/models/shifted_beta_geo.py": {
        "dropout_coefficient": (_CONFIG_KEY, 4),
        "dropout": (
            "name of the sBG likelihood, a different variable from Var.DROPOUT",
            1,
        ),
    },
    "clv/plotting.py": {
        "T": (_LABEL, 1),
    },
    "clv/utils.py": {
        "customer_id": (_DEFAULT, 1),
    },
    "customer_choice/_choice_data_mixin.py": {
        "covariates": (_KEYWORD, 1),
        "depvar": (_KEYWORD, 1),
        "utility_equations": (_KEYWORD, 1),
        "model_config": (_KEYWORD, 1),
        "sampler_config": (_KEYWORD, 1),
    },
    "customer_choice/bayesian_blp.py": {
        "likelihood": ("attrs key of the BLP model", 1),
    },
    "customer_choice/maxdiff.py": {
        "U": (_OWN_MODEL, 1),
        "X": (_OWN_MODEL, 1),
        "model_config": (_KEYWORD, 1),
        "sampler_config": (_KEYWORD, 1),
    },
    "customer_choice/mixed_logit.py": {
        "likelihood": (_CONFIG_KEY, 3),
    },
    "customer_choice/mnl_logit.py": {
        "X": ("name of the instance attribute that holds the data", 1),
        "covariates": (_KEYWORD, 1),
        "depvar": (_KEYWORD, 1),
        "utility_equations": (_KEYWORD, 1),
        "likelihood": (_CONFIG_KEY, 3),
    },
    "customer_choice/mv_its.py": {
        "likelihood": (_CONFIG_KEY, 5),
        "y": (_OWN_MODEL, 4),
        "model_config": (_KEYWORD, 1),
        "sampler_config": (_KEYWORD, 1),
    },
    "customer_choice/nested_logit.py": {
        "covariates": (_KEYWORD, 1),
        "depvar": (_KEYWORD, 1),
        "utility_equations": (_KEYWORD, 1),
        "likelihood": (_CONFIG_KEY, 3),
    },
    "customer_choice/synthetic_data.py": {
        "X": ("key of the synthetic data dictionary", 1),
    },
    "data/fivetran.py": {
        "date": (_DEFAULT, 4),
    },
    "experimental/terms/_graph.py": {
        "mu": (_PARAMETER, 5),
    },
    "mlflow.py": {
        "model_type": ("name of an MLflow parameter", 1),
    },
    "mmm/additive_effect.py": {
        "channel": (_DEFAULT, 2),
        "date": (_DEFAULT, 7),
    },
    "mmm/budget_model.py": {
        "channel": ("column of the lift test or design DataFrame", 7),
    },
    "mmm/budget_optimizer.py": {
        "cost_per_unit": ("label and tensor name of the cost table", 2),
        "date": (_COLUMN + ", or " + _DEFAULT, 4),
        "mu_effects": ("removed keyword that is still recognised", 2),
    },
    "mmm/builders/factories.py": {
        "model_config": ("YAML key", 2),
    },
    "mmm/components/adstock.py": {
        "adstock": (_COMPONENT, 2),
    },
    "mmm/components/saturation.py": {
        "saturation": (_COMPONENT, 2),
    },
    "mmm/counterfactual.py": {
        "time_index": ("label of the batched input tensor", 1),
    },
    "mmm/data_conversion.py": {
        "channel": ("public name of a data variable of the input Dataset", 1),
        "control": ("public name of a data variable of the input Dataset", 1),
    },
    "mmm/estimand_label.py": {
        "induced_spend": ("name of a label field, in the message that names it", 1),
    },
    "mmm/events.py": {
        "saturation": (_COMPONENT, 1),
    },
    "mmm/fourier.py": {
        "fourier_mode": (_DEFAULT, 1),
    },
    "mmm/incrementality.py": {
        "adstock": ("name of the component passed to a check", 2),
        "saturation": ("name of the component passed to a check", 1),
        "allocation": ("estimand option", 4),
        "link": ("key of the reported assumptions", 1),
        "time_varying_media": ("key of the reported assumptions", 1),
        "induced_spend": ("name of the argument, in the message that refuses it", 2),
    },
    "mmm/lift_test.py": {
        "channel": (_COLUMN, 1),
        "date": (_COLUMN, 2),
        "mu": (_PARAMETER, 1),
    },
    "mmm/media_transformation.py": {
        "adstock": ("field of the serialized media transformation", 2),
        "saturation": ("field of the serialized media transformation", 2),
    },
    "mmm/mmm.py": {
        "adstock": (_KEYWORD, 1),
        "saturation": (_KEYWORD, 1),
        "cost_per_unit": (_KEYWORD, 1),
        "link": (_KEYWORD, 1),
        "model_config": (_KEYWORD, 1),
        "sampler_config": (_KEYWORD, 1),
        "time_varying_intercept": (_KEYWORD, 1),
        "time_varying_media": (_KEYWORD, 1),
        "channel": (_COLUMN, 13),
        "date": (_COLUMN + ", or " + _DEFAULT, 13),
        "y": (_DEFAULT, 1),
    },
    "mmm/optimization_variables.py": {
        "channel": (_DEFAULT, 1),
        "date": (_DEFAULT, 3),
    },
    "mmm/plot.py": {
        "channel": (_LABEL + ", or " + _DEFAULT, 4),
        "date": (_COLUMN + ", or " + _LABEL, 8),
        "y_original_scale": (_LABEL, 1),
    },
    "mmm/plot_interactive.py": {
        "channel": (_DEFAULT, 4),
        "channel_scale": (_SCRATCH, 3),
    },
    "mmm/plotting/cv.py": {
        "cv": (_DEFAULT, 1),
    },
    "mmm/plotting/decomposition.py": {
        "y": ("matplotlib keyword a caller may not set", 1),
    },
    "mmm/plotting/diagnostics.py": {
        "y": (_SCRATCH, 1),
    },
    "mmm/plotting/transformations.py": {
        "y": (_SCRATCH, 3),
    },
    "mmm/scaling.py": {
        "channel": ("field of the serialized scaling config", 3),
    },
    "mmm/sensitivity_analysis.py": {
        "sweep": ("name of the symbolic sweep tensor", 1),
    },
    "mmm/summary/cv.py": {
        "date": (_COLUMN, 5),
    },
    "mmm/summary/factory.py": {
        "channel": (_COLUMN, 2),
        "control": (_COLUMN, 1),
        "y": (_SCRATCH, 1),
    },
    "mmm/synthetic_data/endogenous_spend.py": {
        "channel": ("column of the synthetic data", 2),
        "date": ("column of the synthetic data", 1),
        "y": ("column of the synthetic data", 1),
    },
    "mmm/utils.py": {
        "adstock": ("name of an MMM attribute in a check", 1),
    },
    "mmm/var_baseline.py": {
        "channel": ("word in a warning message", 1),
        "control": ("word in a warning message", 1),
    },
    "model_builder.py": {
        "model_config": (_KEYWORD, 1),
        "sampler_config": (_KEYWORD, 1),
    },
    "pie/model.py": {
        "model_config": (_KEYWORD, 1),
    },
}

ATTRIBUTE_NAME_FUNCTIONS = {"getattr", "hasattr", "setattr", "delattr"}
# Names classes whose values are dictionary keys and never attributes. Only their
# values may be typed as the attribute name in ``getattr`` and the like.
DICTIONARY_KEY_CLASSES = {"Attr", "Key"}
NAMES_PACKAGE = "pymc_marketing.names"


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
        if isinstance(node, ast.Import):
            # ``import pymc_marketing.names.mmm as nm``
            for alias in node.names:
                if alias.name == NAMES_PACKAGE or alias.name.startswith(
                    f"{NAMES_PACKAGE}."
                ):
                    submodule = alias.name[len(NAMES_PACKAGE) + 1 :]
                    if submodule in SCOPES.values():
                        modules.add(submodule)
                    if alias.asname is not None:
                        local_names.add(alias.asname)
            continue
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == "pymc_marketing":
            local_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "names"
            )
        elif node.module == NAMES_PACKAGE:
            for alias in node.names:
                if alias.name in SCOPES.values():
                    modules.add(alias.name)
                    local_names.add(alias.asname or alias.name)
                elif alias.name in identifiers:
                    local_names.add(alias.asname or alias.name)
        elif node.module.startswith(f"{NAMES_PACKAGE}."):
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


def _dotted(node: ast.AST) -> str | None:
    """Return ``"a.b.c"`` for the attribute chain ``a.b.c``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def _unannotated_uses(tree: ast.AST, imported: set[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(line, function name)`` for names constants in unannotated functions."""

    def reaches_names(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in imported
        # ``import pymc_marketing.names.mmm`` without an alias
        return isinstance(node, ast.Attribute) and _dotted(node) == NAMES_PACKAGE

    def visit(
        node: ast.AST, function: ast.FunctionDef | ast.AsyncFunctionDef | None
    ) -> Iterator[tuple[int, str]]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node
        elif (
            function is not None and not _is_annotated(function) and reaches_names(node)
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
    used: dict[str, list[int]] = {}
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
            if text in allowed:
                used.setdefault(text, []).append(node.lineno)
                continue
            owner, constant = values[text][0]
            violations.append(
                f"{location}:{node.lineno}: literal {text!r} is "
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
    for text, lines in sorted(used.items()):
        expected = allowed[text][1]
        if len(lines) != expected:
            violations.append(
                f"{location}: {len(lines)} literal(s) {text!r} (lines "
                f"{', '.join(map(str, sorted(lines)))}), but ALLOWED lists "
                f"{expected}. Use the constant for a new one, or update the count"
            )
    return violations, set(used)


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
            "annotate the function, or add an ALLOWED entry for the file with a reason "
            "and a count."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
