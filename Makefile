#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = pymc_marketing

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init lint check_lint format check_format test html fasthtml cleandocs run_notebooks uml help

init: ## Install the package in editable mode
	python3 -m pip install -e .

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

html: ## Install documentation dependencies and build HTML docs (full build)
	pip install .[docs]
	python scripts/generate_gallery.py
	sphinx-build docs/source docs/build -b html

fasthtml: ## Build HTML docs in FAST mode (skip notebooks and heavy API, ~30-60 sec)
	@echo "======================================================================"
	@echo "⚡ FAST BUILD MODE - Development Documentation Build"
	@echo "======================================================================"
	@echo "Building documentation without notebooks and API generation..."
	@echo "For full build with all notebooks, use: make html"
	@echo "======================================================================"
	pip install .[docs]
	python scripts/generate_gallery.py
	PYMC_MARKETING_FAST_DOCS=1 sphinx-build docs/source docs/build -b html
	@echo "======================================================================"
	@echo "✓ Fast build complete! Open docs/build/index.html"
	@echo "======================================================================"

fasthtml-nb: ## Build HTML docs skipping only notebooks (API included)
	@echo "======================================================================"
	@echo "⚡ PARTIAL FAST BUILD - Skipping Notebooks Only"
	@echo "======================================================================"
	pip install .[docs]
	python scripts/generate_gallery.py
	SKIP_NOTEBOOKS=1 sphinx-build docs/source docs/build -b html
	@echo "======================================================================"
	@echo "✓ Build complete (notebooks skipped)!"
	@echo "======================================================================"

fasthtml-api: ## Build HTML docs skipping only API generation (notebooks included)
	@echo "======================================================================"
	@echo "⚡ PARTIAL FAST BUILD - Skipping API Generation Only"
	@echo "======================================================================"
	pip install .[docs]
	python scripts/generate_gallery.py
	SKIP_API_GENERATION=1 sphinx-build docs/source docs/build -b html
	@echo "======================================================================"
	@echo "✓ Build complete (API generation skipped)!"
	@echo "======================================================================"

cleandocs: ## Clean the documentation build directories
	@echo "Cleaning documentation build artifacts..."
	rm -rf docs/build docs/jupyter_execute docs/source/api/generated docs/source/.jupyter_cache
	@echo "✓ Documentation directories cleaned!"

cleancache: ## Clean only the jupyter cache (keeps built docs)
	@echo "Cleaning Jupyter notebook cache..."
	rm -rf docs/source/.jupyter_cache docs/jupyter_execute
	@echo "✓ Jupyter cache cleaned!"

run_notebooks: ## Run Jupyter notebooks
	python scripts/run_notebooks/runner.py

run_notebooks_mmm: ## Run MMM Jupyter notebooks only
	python scripts/run_notebooks/runner.py --exclude-dirs clv bass customer_choice general

run_notebooks_other: ## Run non-MMM Jupyter notebooks
	python scripts/run_notebooks/runner.py --exclude-dirs mmm

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
