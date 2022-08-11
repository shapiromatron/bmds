.PHONY: clean clean-test clean-pyc clean-build lint format docs release dist test test-mpl-regenerate
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

dev: ## Start developer environment
	./bin/dev.sh

clean: clean-build clean-pyc clean-test ## remove all build, test and Python artifacts

clean-build: ## remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -fr htmlcov/

lint:  ## Check for python formatting issues via black & flake8
	@black . --check && isort -q --check . && flake8 .

format:  ## Modify python code using black & show flake8 issues
	@black . && isort -q . && flake8 .

test:
	py.test

test-mpl-regenerate:
	py.test --mpl-generate-path=tests/data/mpl

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

release: dist ## package and upload a release
	twine upload dist/*
	git tag -a "$(shell python setup.py --version)" -m ""
	git push --tags

dist: clean ## builds source and wheel package
	python setup.py bdist_wheel
	ls -l dist
