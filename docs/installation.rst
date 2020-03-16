Installing
==========

A few tips and tricks are required to install this package, due to issues
installing the scientific python stack on Windows, and BMDS on Mac/Linux.

.. _windows-install:

Installing on Windows
~~~~~~~~~~~~~~~~~~~~~

Requires Python 3.6+. When installing on windows, using Anaconda_ is recommended for simplicity,
but using a standard install within Python virtual environment will also work.

.. _Anaconda: https://www.continuum.io/

.. _mac-install:

Installing on Mac/linux
~~~~~~~~~~~~~~~~~~~~~~~

Due to limitations in the underlying BMDS software, the BMDS will only
execute on Windows. However, by using a (remote) Windows server which executes
BMDS, you can run this software on any platform. Install the package as you
normally would, but in addition, you'll need to add a few additional environment
variables to specify the `BMDS server`_  URL and login credentials:

.. code-block:: bash

    export "BMDS_REQUEST_URL=http://bmds-server.com/api/dfile/"
    export "BMDS_TOKEN=abcdefghijklmnopqrstuvwxyz"

.. note::

    Need access to a BMDS server? Want to try mine? Sure, just `ping me!`_

.. _`BMDS server`: https://github.com/shapiromatron/bmds-server
.. _`ping me!`: mailto:shapiromatron@gmail.com

Installing a development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install a development version, first checkout from git::

    git clone https://github.com/shapiromatron/bmds

Change paths to the newly created ``bmds`` folder. Then, preferably in a
python virtual environment, run the command::

    pip install -e .[dev,test]

Tests are written using `pytest`_. To run all tests::

    py.test

.. _`pytest`: http://doc.pytest.org/en/latest/

To run a specific test, use::

    py.test -k test_my_special_test_name

Some tests also create export files, which can be manually inspected to ensure
that they're in the proper format. By default, tests do not create manual
export files, though all other aspects of the API are executed to ensure that
they method calls work as expected. To enable creation of the output files for
inspection, set the environment variable ``BMDS_CREATE_OUTPUTS=TRUE``.

There's a built in Makefile command, ``make dev`` that creates a tmux
application which auto-update the documentation; check out the ``Makefile`` for
a list of other built-in actions.
