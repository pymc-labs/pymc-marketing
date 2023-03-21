.PHONY: init lint check_lint test html cleandocs

init:
	python3 -m pip install -e .

lint:
	pip install .[lint]
	isort .
	black .
	pyreverse pymc_marketing/mmm -d docs/source/uml -f 'ALL' -o png -p mmm
	pyreverse pymc_marketing/clv -d docs/source/uml -f 'ALL' -o png -p clv

check_lint:
	pip install .[lint]
	flake8 .
	isort --check-only .
	black --diff --check --fast .

test:
	pip install .[test]
	pytest

html:
	pip install .[docs]
	sphinx-build docs/source docs/build -b html

cleandocs:
	rm -r "docs/build" "docs/jupyter_execute" "docs/source/api/generated"
