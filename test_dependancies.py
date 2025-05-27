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
    print(f"✓ pymc_marketing importable")
    
    from pymc_marketing.mmm.multidimensional import MMM
    print(f"✓ MMM importable from multidimensional module")
    
    import pydantic
    print(f"✓ pydantic version: {pydantic.__version__}")
    
    from pydantic import ValidationError, validate_call
    print(f"✓ pydantic decorators importable")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\nAll dependencies installed successfully!")