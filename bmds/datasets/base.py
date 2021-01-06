from typing import List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel

from ..constants import Dtype


class DatasetBase:
    # Abstract parent-class for dataset-types.

    dtype: Dtype

    def _validate(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def as_dfile(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def plot(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def drop_dose(self):
        raise NotImplementedError("Abstract method; requires implementation")

    @property
    def num_dose_groups(self):
        return len(set(self.doses))

    def to_dict(self):
        return self.serialize().dict()

    @property
    def dose_linspace(self) -> np.ndarray:
        if not hasattr(self, "_dose_linspace"):
            self._dose_linspace = np.linspace(np.min(self.doses), np.max(self.doses), 100)
        return self._dose_linspace

    def _get_dose_units_text(self):
        if "dose_units" in self.metadata:
            return f" ({self.metadata['dose_units']})"
        return ""

    def _get_response_units_text(self):
        if "response_units" in self.metadata:
            return f" ({self.metadata['response_units']})"
        return ""

    def _get_dataset_name(self):
        return self.metadata.get("dataset_name", "BMDS output results")

    def serialize(self) -> "DatasetSchemaBase":
        raise NotImplementedError("Abstract method; requires implementation")


DatasetType = TypeVar("DatasetType", bound=DatasetBase)  # noqa


class DatasetSchemaBase(BaseModel):
    pass

    def deserialize(self) -> DatasetType:
        raise NotImplementedError("Abstract method; requires implementation")


class DatasetMetadata(BaseModel):
    id: Optional[int]
    name: Optional[str]
    dose_units: Optional[str]
    response_units: Optional[str]
    dose_name: Optional[str]
    response_name: Optional[str]


class DatasetPlottingSchema(BaseModel):
    mean: Optional[List[float]]
    ll: List[float]
    ul: List[float]
