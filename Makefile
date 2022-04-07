.PHONY: init lint check_lint test

init:
	python -m pip install -e .

lint:
	pip install -r lint-requirements.txt
	isort .
	black .

check_lint:
	pip install -r lint-requirements.txt
	flake8 .
	isort --check-only .
	black --diff --check --fast .

test:
	pip install -r test-requirements.txt
	pytest
