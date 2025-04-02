#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = pymc_marketing

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init lint check_lint format check_format test html cleandocs run_notebooks uml help clean-install

init: ## Install the package in editable mode
	python3 -m pip install -e .

clean-install: ## Clean pip cache and install dependencies with force-reinstall
	python3 -m pip cache purge
	python3 -m pip install --upgrade pip setuptools wheel
	python3 -m pip install -e . --force-reinstall --no-cache-dir

lint: ## Install linting dependencies and run linter (ruff and mypy)
	pip install .[lint]
	ruff check $(PACKAGE_DIR) --fix
	mypy .

check_lint: ## Install linting dependencies and check linting (ruff and mypy)
	pip install .[lint]
	ruff check $(PACKAGE_DIR)
	mypy .

format: ## Install linting dependencies and format code (ruff)
	pip install .[lint]
	ruff format $(PACKAGE_DIR)

check_format: ## Install linting dependencies and check code formatting (ruff)
	pip install .[lint]
	ruff format --check $(PACKAGE_DIR)

test:  ## Install test dependencies and run tests
	pip install .[test]
	pytest

html: ## Install documentation dependencies and build HTML docs
	pip install --upgrade pip setuptools wheel
	pip install osqp --no-binary osqp
	pip install .[docs]
	python scripts/generate_gallery.py
	sphinx-build docs/source docs/build -b html

cleandocs: ## Clean the documentation build directories
	rm -r "docs/build" "docs/jupyter_execute" "docs/source/api/generated"

run_notebooks: ## Run Jupyter notebooks
	python scripts/run_notebooks/runner.py

uml: ## Install documentation dependencies and generate UML diagrams
	pip install .[docs]
	pyreverse pymc_marketing/mmm -d docs/source/uml -f 'ALL' -o png -p mmm
	pyreverse pymc_marketing/clv -d docs/source/uml -f 'ALL' -o png -p clv
	pyreverse pymc_marketing/customer_choice -d docs/source/uml -f 'ALL' -o png -p customer_choice

mlflow_server: ## Start MLflow server on port 5000
	mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
