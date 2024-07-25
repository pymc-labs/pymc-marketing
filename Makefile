.PHONY: init lint check_lint format check_format test html cleandocs run_notebooks uml

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

uml:
	pip install .[docs]
	pyreverse pymc_marketing/mmm -d docs/source/uml -f 'ALL' -o png -p mmm
	pyreverse pymc_marketing/clv -d docs/source/uml -f 'ALL' -o png -p clv
