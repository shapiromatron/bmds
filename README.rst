BMDS Python interface
=====================

.. image:: https://img.shields.io/pypi/v/bmds.svg
        :target: https://pypi.python.org/pypi/bmds

.. image:: https://github.com/shapiromatron/bmds/workflows/CI/badge.svg
        :target: https://github.com/shapiromatron/bmds/actions

.. image:: https://readthedocs.org/projects/bmds/badge/?version=latest
        :target: https://bmds.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/61229626.svg
   :target: https://zenodo.org/badge/latestdoi/61229626

This Python package is designed to run the `USEPA BMDS`_ software from a python
interface. It requires Python3.11+.

.. _`USEPA BMDS`: https://www.epa.gov/bmds


(TODO - remove this block and document in bmds-core)
To generate type stubs for compiled code::

    pip install -U mypy
    cd ~/dev/bmds/bmds/
    stubgen -p bmdscore -o .
    ruff . --fix
