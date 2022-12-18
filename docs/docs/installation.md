# Installation

To install the latest official release:

```
pip install bmds
```

## Development

To install a development version (instructions for Mac/Linux; may need to be adapted for Windows):

```bash
# clone repo
git clone https://github.com/shapiromatron/bmds

# create a new python virtual environment
python -m venv venv

# active virtual environment
source venv/bin/activate

# install package in developer mode and developer tools
pip install -r requirements_dev.txt
```

Tests are written using [pytest](http://doc.pytest.org/en/latest/). To run all tests:

```bash
# run all tests
py.test

# To run a specific test
py.test -k test_my_special_test_name
```

There's a built in Makefile command, ``make dev`` that creates a tmux application which auto-update the documentation; check out the ``Makefile`` for a list of other built-in actions.

## BMDS Priors Report

The Python BMDS library includes bayesian priors and frequentist parameter initialization settings that have been tuned to help improve model fit performance. To generate a report of the settings in all the possible permutations, run the command:

```bash
bmds-priors-report priors_report.md
```
