.. _performance:

Performance
~~~~~~~~~~~

In a hurry? Want more speed 🏁🏁🏁? Or want to run a microarray? Let's
speed things up. Start with a JSON dataset that has data that looks like this,
except with lots more data or it wouldn't be worth it:

.. code-block:: json

    [
        "datasets": [
            {
                "id": 6,
                "doses": [
                    0, 0, 0.1, 0.1, 1, 1, 10,
                    10, 100, 100, 300, 300, 500, 500
                ],
                "responses": [
                    8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509,
                    9.1556, 9.6821, 9.8256, 10.2095, 10.2222, 12.0382, 11.0567
                ]
            }
        ]
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
