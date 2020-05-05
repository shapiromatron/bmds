API
===

The section documents some of the key objects and methods used in the
BMDS module.

BMDS session
------------

A BMDS sessions is the primary input for BMDS. It generally contains a dataset,
a collection of models, and optionally logic on which model was recommended as
a best fitting model. Each BMDS session object is related to a specific BMDS
model version.  To get the latest version BMDS Session, you can use the :func:`bmds.session.BMDS.latest_version` method.

.. autoclass:: bmds.session.BMDS
    :members:

.. autoclass:: bmds.session.BMDS_v231
    :members:

.. autoclass:: bmds.session.BMDS_v240
    :members:

.. autoclass:: bmds.session.BMDS_v260
    :members:

.. autoclass:: bmds.session.BMDS_v2601
    :members:

.. autoclass:: bmds.session.BMDS_v270
    :members:

.. autoclass:: bmds.session.BMDS_v312
    :members:

Datasets
--------

A single dataset object is required for each BMD session. There are three
dataset types implemented currently:

- :class:`bmds.datasets.DichotomousDataset`
- :class:`bmds.datasets.DichotomousCancerDataset`
- :class:`bmds.datasets.ContinuousDataset`
- :class:`bmds.datasets.ContinuousIndividualDataset`

.. autoclass:: bmds.datasets.DichotomousDataset
    :members:

.. autoclass:: bmds.datasets.DichotomousCancerDataset
    :members:

.. autoclass:: bmds.datasets.ContinuousDataset
    :members:

.. autoclass:: bmds.datasets.ContinuousIndividualDataset
    :members:

BMD models
----------

.. autoclass:: bmds.models.BMDModel
    :members:

Batch runs using SessionBatch
-----------------------------

.. automodule:: bmds.batch
    :members:

Reporting in Word
-----------------

Sending BMDS results from a session is also possible.

To save a single Session to a Word file, use the
:func:`bmds.session.BMDS.to_docx` method.

To save multiple sessions to a Word file, use a :class:`bmds.reporter.Reporter`
object. This also allows a custom Word template and custom Word styles to be
applied to outputs.

.. autoclass:: bmds.reporter.Reporter
    :members: __init__, add_session, save
