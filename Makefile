.PHONY: init lint check_lint test html cleandocs

init:
	python3 -m pip install -e .

lint:
	pip install .[lint]
	ruff . --fix
	mypy .

check_lint:
	pip install .[lint]
	ruff .
	mypy .

test:
	pip install .[test]
	pytest

html:
	pip install .[docs]
	sphinx-build docs/source docs/build -b html

cleandocs:
	rm -r "docs/build" "docs/jupyter_execute" "docs/source/api/generated"
