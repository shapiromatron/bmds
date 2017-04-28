.. _performance:

Performance
~~~~~~~~~~~

In a hurry? Want more speed 🏁🏁🏁? Or want to run a microarray? Let's
speed things up. Start with a JSON dataset that has data that looks like this,
except with lots more data or it wouldn't be worth it:

.. code-block:: json

   [
    {
        "id": 6,
        "doses": [0.0, 1340.0, 2820.0, 5600.0, 11125.0, 23000.0],
        "ns": [10, 10, 10, 10, 10, 8],
        "means": [29.3, 28.4, 27.2, 26.0, 25.8, 24.5],
        "stdevs": [2.53, 1.9,  2.53, 2.53, 2.21, 1.58]
    },
    {
        "id": 9,
        "doses": [0.0, 20.0, 40.0, 80.0, 170.0, 320.0],
        "ns": [10, 10, 10, 10, 10, 10],
        "means": [35.79, 36.69, 37.84, 44.2, 48.03, 58.4],
        "stdevs": [0.56, 0.36, 0.51, 0.27, 0.89, 1.4]
    }
  ]

Then, create a script like this for parallel execution. Note that you can set
the ``base_session`` here to define custom models or recommendation logic,
or you could even use a method to change the session on the fly; this example
script runs all default continuous models with the default BMR:

.. literalinclude:: performance.py
   :language: python

Run the script as follows (assuming the script above is named
``performance.py`` and the input dataset is ingeniously named ``input.json``):

.. code-block:: batch

    python performance.py C:\path\to\input.json C:\path\to\output.json
