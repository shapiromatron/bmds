from typing import TypeVar

from .base import DatasetBase  # noqa
from .continuous import ContinuousDataset, ContinuousIndividualDataset  # noqa
from .dichotomous import DichotomousCancerDataset, DichotomousDataset  # noqa

DatasetType = TypeVar("DatasetType", bound=DatasetBase)  # noqa