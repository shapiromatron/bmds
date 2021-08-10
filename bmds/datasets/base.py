import abc
from typing import Dict, List, Optional, TypeVar

import numpy as np
from pydantic import BaseModel

from ..constants import ZEROISH, Dtype


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


class DatasetBase(abc.ABC):
    # Abstract parent-class for dataset-types.

    dtype: Dtype
    metadata: DatasetMetadata

    DEFAULT_XLABEL = "Dose"
    DEFAULT_YLABEL = "Response"

    @abc.abstractmethod
    def _validate(self):
        ...

    @abc.abstractmethod
    def as_dfile(self):
        ...

    @abc.abstractmethod
    def plot(self):
        ...

    @abc.abstractmethod
    def drop_dose(self):
        ...

    @property
    def num_dose_groups(self):
        return len(set(self.doses))

    def to_dict(self):
        return self.serialize().dict()

    @property
    def dose_linspace(self) -> np.ndarray:
        if not hasattr(self, "_dose_linspace"):
            self._dose_linspace = np.linspace(np.min(self.doses), np.max(self.doses), 100)
            self._dose_linspace[self._dose_linspace == 0] = ZEROISH
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

    @abc.abstractmethod
    def serialize(self) -> "DatasetSchemaBase":
        ...

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            dataset_name=self.metadata.name,
            dataset_dose_name=self.metadata.dose_name,
            dataset_dose_units=self.metadata.dose_units,
            dataset_response_name=self.metadata.response_name,
            dataset_response_units=self.metadata.response_units,
        )


DatasetType = TypeVar("DatasetType", bound=DatasetBase)


class DatasetSchemaBase(BaseModel, abc.ABC):
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

    @abc.abstractmethod
    def deserialize(self) -> DatasetBase:
        ...


class DatasetPlottingSchema(BaseModel):
    mean: Optional[List[float]]
    ll: List[float]
    ul: List[float]
