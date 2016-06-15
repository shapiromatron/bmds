BMDS: Benchmark dose modeling software
======================================

This package is designed to run the U.S. EPA BMDS_ software from a python
interface. It is integrated into the HAWC_ software. Example function calls
are shown below:

.. code-block:: python

    import bmds

    # return BMDS versions interface has been implemented for
    bmds.get_versions()

    # get the available BMDS models for the specified version
    bmds.get_models_for_version('2.40')

    # create example datasets
    ds1 = bmds.DichotomousDataset()
    ds2 = bmds.ContinuousDataset()

    # create new model runs
    models = [
        bmds.Gamma_215(ds1),
        bmds.Power_217(ds2),
    ]

    # execute each model and parse results
    for model in models:
        model.execute()
        model.parse_results()


Installation notes:
~~~~~~~~~~~~~~~~~~~

This package should install fine in any platform (Windows, PC, Linux). However,
the BMDS models are only compiled for use in Windows. Therefore, the
``execute`` method is only available in Windows.

.. _BMDS: https://www.epa.gov/bmds
.. _HAWC: https://hawcproject.org
