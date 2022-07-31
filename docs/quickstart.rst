Quickstart
~~~~~~~~~~

Install the software using pip:

.. code-block:: bash

    pip install bmds

An example continuous summary dataset:

.. code-block:: python

    import bmds

    # create a dataset
    dataset = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[25, 25, 24, 24, 24],
        means=[2.61, 2.81, 2.96, 4.66, 11.23],
        stdevs=[0.81, 1.19, 1.37, 1.72, 2.84]
    )

    # create a BMD session
    session = bmds.BMDS.latest_version(
        bmds.constants.CONTINUOUS,
        dataset=dataset)

    # add all default models
    session.add_default_models()

    # execute the session
    session.execute()

    # recommend a best-fitting model
    session.recommend()

    print(session.recommended_model.output['model_name'])
    # 'Exponential-M2'

    print(session.recommended_model.output['BMD'])
    # 93.7803

    # save dose-response plots
    session.save_plots('~/Desktop', format='png')

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


Frequently Asked Questions
--------------------------

1. How can I specify another BMR on ALL default models?

.. code-block:: python

    # get BMR types for dataset type selected
    bmr_types = bmds.constants.BMR_CROSSWALK[bmds.constants.CONTINUOUS]

    # create session
    session = bmds.BMDS.latest_version(
        bmds.constants.CONTINUOUS,
        dataset=dataset)

    # add settings to all models
    settings = {
        'bmr': 0.1,
        'bmr_type': bmr_types['Rel. Dev.']
    }
    session.add_default_models(global_settings=settings)

2. How can I specify other settings on a particular model?

.. code-block:: python

    # get BMR types for dataset type selected
    bmr_types = bmds.constants.BMR_CROSSWALK[bmds.constants.CONTINUOUS]

    # add model and settings
    session.add_model(
        bmds.constants.M_Polynomial,
        settings={
            'constant_variance': 1,
            'degree_poly': 3,
            'bmr': 2.5,
            'bmr_type': bmr_types['Abs. Dev.']
    })

3. How can I modify the default decision logic?

.. code-block:: python

    # create session and add default models
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=dataset)
    session.add_default_models()

    # modify default logic to fail if bmd or bmdl threshold is >=3
    session.add_recommender()
    for rule in session.recommender.rules:
        if isinstance(rule, bmds.rules.LowBmd) or isinstance(rule, bmds.rules.LowBmdl):
            rule.threshold = 3.

    # execute
    session.execute_and_recommend(drop_doses=True)
