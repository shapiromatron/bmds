Quickstart
~~~~~~~~~~

Install the software using pip:

.. code-block:: bash

    pip install bmds

Trouble installing? See notes on :ref:`Windows <windows-install>` or :ref:`Mac/Linux <mac-install>`.

An example continuous summary dataset:

.. code-block:: python

    import bmds

    dataset = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[25, 25, 24, 24, 24],
        means=[2.61, 2.81, 2.96, 4.66, 11.23],
        stdevs=[0.81, 1.19, 1.37, 1.72, 2.84]
    )

    session = bmds.BMDS.latest_version(
        bmds.constants.CONTINUOUS,
        dataset=dataset)
    session.add_default_models()
    session.execute()
    session.recommend()

    # save dose-response plots
    session.save_plots('~/Desktop', format='pdf')

    # save results to an Excel file
    session.to_excel('~/Desktop/results.xlsx')


To use a dichotomous dataset, only a few things change:

.. code-block:: python

    dataset = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14]
    )

    session = bmds.BMDS.latest_version(
        bmds.constants.DICHOTOMOUS,
        dataset=dataset)

To run multiple datasets, you can use a ``SessionBatch``.
