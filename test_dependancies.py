#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Script to check and install required dependencies."""

import subprocess
import sys


def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Core dependencies
required_packages = [
    "pymc>=5.0.0",
    "arviz>=0.13.0",
    "pandas>=1.5.0",
    "numpy>=1.22.0",
    "scipy>=1.9.0",
    "pydantic>=2.0.0",  # Important for validation
    "matplotlib>=3.5.0",
    "seaborn>=0.12.0",
    "xarray>=2022.3.0",
]

# Test dependencies
test_packages = [
    "pytest>=7.0.0",
    "pytest-cov",
    "pytest-mock",
]

print("Installing core dependencies...")
for package in required_packages:
    try:
        install_package(package)
        print(f"✓ {package} installed")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

print("\nInstalling test dependencies...")
for package in test_packages:
    try:
        install_package(package)
        print(f"✓ {package} installed")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

# Verify imports
print("\nVerifying imports...")
try:
    import pymc_marketing

    print("✓ pymc_marketing importable")

    from pymc_marketing.mmm.multidimensional import MMM

    print("✓ MMM importable from multidimensional module")

    import pydantic

    print(f"✓ pydantic version: {pydantic.__version__}")

    from pydantic import ValidationError, validate_call

    print("✓ pydantic decorators importable")

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nAll dependencies installed successfully!")
