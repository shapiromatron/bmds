Quickstart
~~~~~~~~~~

Install the software using pip:

.. code-block:: bash

    pip install bmds

Example function calls shown below:

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

Use on Mac and/or Linux
-----------------------

Due to limitations in the underlying BMDS software, the BMDS will only
execute on Windows. However, by using a (remote) Windows server which executes
BMDS, you can run this software on any platform. Install the package as you
normally would, but in addition, you'll need to add a few additional environment
variables to specify the `BMDS server`_  URL and login credentials:

.. code-block:: bash

    export "BMDS_HOST=http://bmds-server.com"
    export "BMDS_USERNAME=myusername"
    export "BMDS_PASSWORD=mysecret"

That should do it!

.. note::

    These are just example settings, you'll need to configure your own BMDS
    server to use. Want to borrow mine? Sure, just `ping me!`_

.. _`BMDS server`: https://github.com/shapiromatron/bmds-server
.. _`ping me!`: mailto:shapiromatron@gmail.com

