Installing
==========

To install the latest official release:

.. code-block:: bash

    pip install bmds

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

To install a development version (instructions for Mac/Linux; may need to be adapted for Windows):

.. code-block:: bash

    # clone repo
    git clone https://github.com/shapiromatron/bmds

    # create a new python virtual environment
    python -m venv venv

    # active virtual environment
    source venv/bin/activate

    # install package in developer mode and developer tools
    pip install -r requirements_dev.txt

If using Windows, it's recommended to use an `miniconda`_ Python environment for simplicity.

.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html

Tests are written using `pytest`_. To run all tests:

.. code-block:: bash

    # run all tests
    py.test

    # To run a specific test
    py.test -k test_my_special_test_name

.. _`pytest`: http://doc.pytest.org/en/latest/

Some tests also create export files, which can be manually inspected to ensure
that they're in the proper format. By default, tests do not create manual
export files, though all other aspects of the API are executed to ensure that
they method calls work as expected. To enable creation of the output files for
inspection, set the environment variable ``BMDS_CREATE_OUTPUTS=TRUE``.

There's a built in Makefile command, ``make dev`` that creates a tmux
application which auto-update the documentation; check out the ``Makefile`` for
a list of other built-in actions.
