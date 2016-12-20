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
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159]
    )

    session = bmds.BMDS.latest_version(
        bmds.constants.CONTINUOUS,
        dataset=dataset)
    session.add_default_models()
    session.execute()
    session.recommend()

An example dichotomous model execution:

.. code-block:: python

    import bmds

    dataset = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14]
    )

    session = bmds.BMDS.latest_version(
        bmds.constants.DICHOTOMOUS,
        dataset=dataset)
    session.add_default_models()
    session.execute()
    session.recommend()

    print(session.recommended_model.name)
    >>> 'Multistage-2'

    print(session.recommended_model.output['BMD'])
    >>> 18.0607
