import re
from os.path import dirname, join, realpath

from setuptools import find_packages, setup

DESCRIPTION = "Marketing Statistical Models in PyMC"
AUTHOR = "PyMC Labs"
AUTHOR_EMAIL = "info@pymc-labs.io "
URL = "https://github.com/pymc-labs/pymc-marketing"
LICENSE = "Apache License, Version 2.0"

PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

TEST_REQUIREMENTS_FILE = join(PROJECT_ROOT, "test-requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    test_reqs = f.read().splitlines()


def get_version():
    version_file = join("pymc_marketing", "__init__.py")
    lines = open(version_file).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        if mo := re.search(version_regex, line, re.M):
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {version_file}.")


setup(
    name="pymc-marketing",
    version=get_version(),
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_reqs,
    tests_require=test_reqs,
)
