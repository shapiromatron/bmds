Installing
==========

A few tips and tricks are required to install this package, due to issues
installing the scientific python stack on Windows, and BMDS on Mac/Linux.

.. _windows-install:

Installing on Windows
~~~~~~~~~~~~~~~~~~~~~

Installing scientific python numpy_ and scipy_ can be tricky on Windows, and
both are required for using the BMDS package. Here are two options, and recommended steps:

1. Use Anaconda_.
2. Use Python 3.6 and virtual environments. If you choose this method:
    - Download the appropriate `numpy and scipy binaries`_
    - Create a new virtual environment
    - In the virtual environment, run the commands (with proper paths)::

        pip install C:\Users\...\Downloads\numpy.whl
        pip install C:\Users\...\Downloads\scipy.whl
        pip install bmds

.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _Anaconda: https://www.continuum.io/
.. _`numpy and scipy binaries`: http://www.lfd.uci.edu/~gohlke/pythonlibs/

.. _mac-install:

Installing on Mac/linux
~~~~~~~~~~~~~~~~~~~~~~~

Due to limitations in the underlying BMDS software, the BMDS will only
execute on Windows. However, by using a (remote) Windows server which executes
BMDS, you can run this software on any platform. Install the package as you
normally would, but in addition, you'll need to add a few additional environment
variables to specify the `BMDS server`_  URL and login credentials:

.. code-block:: bash

    export "BMDS_HOST=http://bmds-server.com"
    export "BMDS_USERNAME=myusername"
    export "BMDS_PASSWORD=mysecret"

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
