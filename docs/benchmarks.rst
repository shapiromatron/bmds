Benchmarks
==========

This p


The `benchmarks/toxrefdb` folder creates a test application which executes models and store the results in the database app.db

The current test application can be repeated on future versions of bmds.

Then entrypoint for the testing application is in `benchmarks/main.py`. run_analysis methods executes the test application. below is the example command to execute the run_analysis method

To install the additional dependencies required to run benchmarks, assuming you've setup the developer environment as previously described:

.. code-block:: bash

   pip install -e .[benchmarks]


An example analysis


.. code-block:: python

   from bmds import benchmarks

   # first time-only
   Benchmark.create_db()

   # load dataset and execute
   Benchmark.load_dataset(Dataset.TOXREFDB_DICH, "dichotomous_tr.csv")

   # run multiple versions
   Benchmark.run_analysis(Dataset.TOXREFDB_DICH, [Versions.BMDS270, Versions.BMDS330], True)

