Install a development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install a development version, first checkout from git::

    git clone https://github.com/shapiromatron/bmds

Change paths to the newly created ``bmds`` folder. Then, preferably in a
python virtual environment, run the command::

    pip install -r requirements.txt
    pip install -e .

Tests are written using `pytest`_. To run all tests::

    py.test

.. _`pytest`: http://doc.pytest.org/en/latest/

To run a specific test, use::

    py.test -k test_my_special_test_name

There's a built in Makefile command, ``make dev`` that creates a tmux
application which auto-update the documentation; check out the ``Makefile`` for
a list of other built-in actions.
