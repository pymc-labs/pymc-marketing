.PHONY: init lint check_lint test html cleandocs run_notebooks

PACKAGE_DIR = pymc_marketing

init:
	python3 -m pip install -e .

lint:
	pip install .[lint]
	ruff check $(PACKAGE_DIR) --fix
	mypy .

check_lint:
	pip install .[lint]
	ruff check $(PACKAGE_DIR)
	mypy .

format:
	pip install .[lint]
	ruff format $(PACKAGE_DIR)

check_format:
	pip install .[lint]
	ruff format --check $(PACKAGE_DIR)

test:
	pip install .[test]
	pytest

html:
	pip install .[docs]
	sphinx-build docs/source docs/build -b html

cleandocs:
	rm -r "docs/build" "docs/jupyter_execute" "docs/source/api/generated"

run_notebooks:
	python scripts/run_notebooks/runner.py