API
===

The section documents some of the key objects and methods used in the
BMDS module.

BMDS session
------------

.. automodule:: bmds.session
    :members:


Datasets
--------

A single dataset object is required for each BMD session. There are three
dataset types implemented currently:

- :class:`bmds.datasets.DichotomousDataset`
- :class:`bmds.datasets.ContinuousDataset`
- :class:`bmds.datasets.ContinuousIndividualDataset`

.. autoclass:: bmds.datasets.DichotomousDataset
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
