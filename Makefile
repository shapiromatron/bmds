.PHONY: dev docs docs-serve lint format test test-mpl test-mpl-regenerate clean dist release loc
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

docs: ## Build documentation
	cd docs; mkdocs build --strict

docs-serve: ## Generate documentation
	cd docs; mkdocs serve -a localhost:8050

lint: ## Check formatting issues
	@black . --check && ruff .

format: ## Fix formatting issues where possible
	@black . && ruff . --fix --show-fixes

test: ## Run all tests, except matplotlib figures
	py.test

test-mpl: ## Run all tests; compare matplotlib figures
	py.test --mpl

test-mpl-regenerate: ## Regenerate matplotlib figures in tests
	py.test --mpl-generate-path=tests/data/mpl

clean: ## Remove all build, test and Python artifacts
	# remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	# remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	# remove test and coverage artifacts
	rm -fr .tox/
	rm -fr htmlcov/

dist: clean ## Builds python wheels
	python setup.py bdist_wheel
	ls -l dist

release: dist ## Package and upload a release
	twine upload dist/*
	git tag -a "$(shell python setup.py --version)" -m ""
	git push --tags

loc: ## Generate lines of code report
	@cloc \
		--exclude-dir=build,dist,notebooks,venv \
		--exclude-ext=json,yaml,svg,toml,ini \
		--vcs=git \
		--counted loc-files.txt \
		--md \
		--report-file=loc.txt \
		.
