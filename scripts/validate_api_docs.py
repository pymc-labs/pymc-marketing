#!/usr/bin/env python3
"""Validate that all public modules are documented in API docs.

This script checks that docs/source/api/index.md includes all public
modules from pymc_marketing/, excluding intentionally internal modules.

Usage:
    python scripts/validate_api_docs.py [--verbose] [--fix]

Options:
    --verbose, -v    Show detailed list of included and excluded modules
    --fix            Automatically add missing modules to the API docs

Exit codes:
    0 - All modules documented
    1 - Missing modules or other validation errors
"""

import argparse
import re
import sys
from pathlib import Path

# Modules to exclude from documentation (internal utilities)
EXCLUDE_MODULES = {
    "__init__",
    "version",
    "py",  # py.typed
    "constants",  # Internal constants only
    "decorators",  # Internal decorators only
}


def get_package_modules(package_dir: Path) -> set[str]:
    """Get all public modules from the pymc_marketing package.

    Parameters
    ----------
    package_dir : Path
        Path to the pymc_marketing directory.

    Returns
    -------
    set[str]
        Set of module names (without .py extension).
    """
    modules = set()

    if not package_dir.exists():
        print(f"❌ Error: Package directory not found: {package_dir}")
        sys.exit(1)

    # Find all .py files (modules)
    for item in package_dir.iterdir():
        if item.is_file() and item.suffix == ".py":
            module_name = item.stem
            if module_name not in EXCLUDE_MODULES:
                modules.add(module_name)

        # Find all subdirectories with __init__.py (subpackages)
        elif item.is_dir() and not item.name.startswith("__"):
            init_file = item / "__init__.py"
            if init_file.exists():
                modules.add(item.name)

    return modules


def get_documented_modules(api_index_file: Path) -> set[str]:
    """Extract documented modules from docs/source/api/index.md.

    Parameters
    ----------
    api_index_file : Path
        Path to the API index markdown file.

    Returns
    -------
    set[str]
        Set of documented module names.
    """
    if not api_index_file.exists():
        print(f"❌ Error: API index file not found: {api_index_file}")
        sys.exit(1)

    content = api_index_file.read_text()

    # Find the autosummary section
    # Pattern: lines between autosummary directive and closing ```
    pattern = r"```{eval-rst}.*?autosummary:.*?\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("❌ Error: Could not find autosummary section in API index")
        sys.exit(1)

    autosummary_content = match.group(1)

    # Extract module names (non-indented lines that aren't directives)
    modules = set()
    for line in autosummary_content.split("\n"):
        line = line.strip()
        # Skip empty lines, directives (start with :), and options
        if line and not line.startswith(":") and not line.startswith(".."):
            modules.add(line)

    return modules


def fix_api_docs(
    api_index_file: Path, documented_modules: set[str], missing_modules: set[str]
) -> bool:
    """Add missing modules to the API documentation file.

    Parameters
    ----------
    api_index_file : Path
        Path to the API index markdown file.
    documented_modules : set[str]
        Currently documented modules.
    missing_modules : set[str]
        Modules that need to be added.

    Returns
    -------
    bool
        True if file was modified, False otherwise.
    """
    if not missing_modules:
        return False

    # Read the current content
    content = api_index_file.read_text()

    # Find the autosummary section - capture everything before, the section, and after
    pattern = r"(.*?```{eval-rst}.*?autosummary:.*?\n)(.*?)(```.*)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("❌ Error: Could not find autosummary section to update")
        return False

    before_autosummary = match.group(1)
    autosummary_content = match.group(2)
    after_closing = match.group(3)

    # Parse the current module list (preserving directives and formatting)
    lines = autosummary_content.split("\n")
    directive_lines = []
    module_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(":") or stripped.startswith(".."):
            directive_lines.append(line)
        else:
            module_lines.append(stripped)

    # Add missing modules and sort
    all_modules = sorted(set(module_lines) | missing_modules)

    # Reconstruct the content - preserve blank lines from directives
    new_autosummary = "\n".join(directive_lines)
    # Only add separator if there isn't already a trailing newline
    if new_autosummary and not new_autosummary.endswith("\n"):
        new_autosummary += "\n"
    new_autosummary += "\n".join(f"  {module}" for module in all_modules)
    new_autosummary += "\n"

    new_content = before_autosummary + new_autosummary + after_closing

    # Write back
    api_index_file.write_text(new_content)

    return True


def main() -> int:
    """Run validation and return exit code."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Validate that all public modules are documented in API docs."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed list of included and excluded modules",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically add missing modules to docs/source/api/index.md",
    )
    args = parser.parse_args()

    # Determine repository root (script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    package_dir = repo_root / "pymc_marketing"
    api_index_file = repo_root / "docs" / "source" / "api" / "index.md"

    print("🔍 Validating API documentation completeness...")
    print()

    # Get modules from package
    package_modules = get_package_modules(package_dir)

    # Get documented modules
    documented_modules = get_documented_modules(api_index_file)

    # Get all modules that were found in the package (before exclusion)
    all_found_modules = set()
    for item in package_dir.iterdir():
        if item.is_file() and item.suffix == ".py":
            all_found_modules.add(item.stem)
        elif item.is_dir() and not item.name.startswith("__"):
            init_file = item / "__init__.py"
            if init_file.exists():
                all_found_modules.add(item.name)

    excluded_found_modules = all_found_modules & EXCLUDE_MODULES

    # Find missing and phantom modules
    missing_modules = package_modules - documented_modules
    phantom_modules = documented_modules - package_modules

    # Handle --fix flag for missing modules
    if args.fix and missing_modules:
        print("🔧 Auto-fixing: Adding missing modules to docs/source/api/index.md...")
        print()
        for module in sorted(missing_modules):
            print(f"   + {module}")
        print()

        if fix_api_docs(api_index_file, documented_modules, missing_modules):
            print("✅ Successfully updated docs/source/api/index.md")
            print()

            # Re-validate to confirm
            documented_modules = get_documented_modules(api_index_file)
            missing_modules = package_modules - documented_modules

            if not missing_modules:
                print("✅ Validation now passes!")
                print()
                return 0
            else:
                print("⚠️  Warning: Some modules are still missing after fix")
                print()
        else:
            print("❌ Failed to update API documentation")
            print()
            return 1

    # Report results
    has_errors = False

    if missing_modules:
        has_errors = True
        print(
            "❌ MISSING: The following modules are not documented in docs/source/api/index.md:"
        )
        print()
        for module in sorted(missing_modules):
            print(f"   - {module}")
        print()

        print("📋 How to fix this:")
        print()
        print("  Option 1: Auto-fix (if module should be in public API)")
        print("    python3 scripts/validate_api_docs.py --fix")
        print()
        print(
            "  Option 2: Manually add to docs/source/api/index.md (alphabetically sorted)"
        )
        print(
            "    Edit docs/source/api/index.md and add the module to the autosummary list"
        )
        print()
        print("  Option 3: Exclude module (if it's internal/private)")
        print("    Add module name to EXCLUDE_MODULES in scripts/validate_api_docs.py")
        print()
        print("Example of what docs/source/api/index.md should look like:")
        print()
        all_should_be_documented = sorted(documented_modules | missing_modules)
        for module in all_should_be_documented:
            marker = "  # <-- ADD THIS" if module in missing_modules else ""
            print(f"  {module}{marker}")
        print()

    if phantom_modules:
        has_errors = True
        print("❌ PHANTOM: The following modules are documented but don't exist:")
        print()
        for module in sorted(phantom_modules):
            print(f"   - {module}")
        print()

        print("📋 How to fix this:")
        print()
        print("  Option 1: Remove from documentation (if module was deleted)")
        print("    Edit docs/source/api/index.md and remove these module names")
        print()
        print("  Option 2: Restore the module (if it was accidentally deleted)")
        print("    Add the module file back to pymc_marketing/")
        print()
        print("  Option 3: Fix typo (if module name was misspelled)")
        print("    Edit docs/source/api/index.md and correct the spelling")
        print()

    if has_errors:
        print("❌ API documentation validation FAILED!")
        print()
        if args.verbose:
            print("Excluded modules (internal only):")
            for module in sorted(excluded_found_modules):
                print(f"   - {module}")
        else:
            print("Excluded modules (internal only):", sorted(excluded_found_modules))
        return 1

    # Success
    print("✅ API documentation validation PASSED!")
    print()
    print(f"   - {len(documented_modules)} modules documented")
    print(f"   - {len(excluded_found_modules)} modules excluded (internal)")

    if args.verbose:
        print()
        print("📋 Documented modules (public API):")
        for module in sorted(documented_modules):
            print(f"   ✓ {module}")
        print()
        print("🔒 Excluded modules (internal only):")
        for module in sorted(excluded_found_modules):
            print(f"   ✗ {module}")

    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
