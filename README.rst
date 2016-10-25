BMDS: Benchmark dose modeling software
======================================

.. image:: https://img.shields.io/pypi/v/bmds.svg
        :target: https://pypi.python.org/pypi/bmds

.. image:: https://pypip.in/wheel/bmds/badge.svg
    :target: https://pypi.python.org/pypi/bmds/
    :alt: Wheel Status

.. image:: https://img.shields.io/travis/shapiromatron/bmds.svg
        :target: https://travis-ci.org/shapiromatron/bmds

.. image:: https://readthedocs.org/projects/bmds/badge/?version=latest
        :target: https://bmds.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

This Python 2 package is designed to run the U.S. EPA BMDS_ software from a python
interface. It is integrated into the HAWC_ software, but HAWC does not be installed
on your computer to use this BMDS package. Example function calls are shown below:

.. code-block:: python

    import bmds

    # get available BMDS versions
    bmds.get_versions()

    # get BMDS models for the specified version
    bmds.get_models_for_version('2.40')

    # create example datasets
    ds1 = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14])

    ds2 = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        responses=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])

    # create new model runs
    models = [
        bmds.Gamma_215(ds1),
        bmds.Power_217(ds2),
    ]

    # execute each model and parse results
    for model in models:
        model.execute()


Install a development version:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install a development version, first checkout from git::

    git clone https://github.com/shapiromatron/bmds

Change paths to the newly created ``bmds`` folder. Then, preferably in a
python virtual environment, run the command::

    pip install -r requirements.txt

Check for a successful installation by using the command::

    py.test

This package should install fine in any platform (Windows, PC, Linux). However,
the BMDS models are only compiled for use in Windows. Therefore, the
``execute`` method is only available in Windows.

.. _BMDS: https://www.epa.gov/bmds
.. _HAWC: https://hawcproject.org
