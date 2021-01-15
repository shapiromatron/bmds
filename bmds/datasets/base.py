from typing import Dict, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel

from ..constants import Dtype


class DatasetMetadata(BaseModel):
    id: Optional[int]
    name: str = ""
    dose_units: str = ""
    response_units: str = ""
    dose_name: str = ""
    response_name: str = ""

    class Config:
        extra = "allow"

    def get_name(self):
        if self.name:
            return self.name
        if self.id:
            return f"Dataset #{self.id}"
        return "BMDS output results"


class DatasetBase:
    # Abstract parent-class for dataset-types.

    dtype: Dtype
    metadata: DatasetMetadata

    DEFAULT_XLABEL = "Dose"
    DEFAULT_YLABEL = "Response"

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

    def _get_dose_units_text(self) -> str:
        if self.metadata.dose_units:
            return f" ({self.metadata.dose_units})"
        return ""

    def _get_response_units_text(self) -> str:
        if self.metadata.response_units:
            return f" ({self.metadata.response_units})"
        return ""

    def _get_dataset_name(self) -> str:
        return self.metadata.get_name()

    def get_xlabel(self):
        label = self.DEFAULT_XLABEL
        if self.metadata.dose_name:
            label = self.metadata.dose_name
        if self.metadata.dose_units:
            label += f" ({self.metadata.dose_units})"
        return label

    def get_ylabel(self):
        label = self.DEFAULT_YLABEL
        if self.metadata.response_name:
            label = self.metadata.response_name
        if self.metadata.response_units:
            label += f" ({self.metadata.response_units})"
        return label

    def serialize(self) -> "DatasetSchemaBase":
        raise NotImplementedError("Abstract method; requires implementation")


DatasetType = TypeVar("DatasetType", bound=DatasetBase)


class DatasetSchemaBase(BaseModel):
    @classmethod
    def get_subclass(cls, dtype: Dtype) -> BaseModel:
        from .continuous import ContinuousDatasetSchema, ContinuousIndividualDatasetSchema
        from .dichotomous import DichotomousCancerDatasetSchema, DichotomousDatasetSchema

        _dataset_schema_map: Dict = {
            Dtype.CONTINUOUS: ContinuousDatasetSchema,
            Dtype.CONTINUOUS_INDIVIDUAL: ContinuousIndividualDatasetSchema,
            Dtype.DICHOTOMOUS: DichotomousDatasetSchema,
            Dtype.DICHOTOMOUS_CANCER: DichotomousCancerDatasetSchema,
        }
        try:
            return _dataset_schema_map[dtype]
        except KeyError:
            raise ValueError(f"Unknown dtype: {dtype}")

    def deserialize(self) -> DatasetBase:
        raise NotImplementedError("")


class DatasetPlottingSchema(BaseModel):
    mean: Optional[List[float]]
    ll: List[float]
    ul: List[float]
