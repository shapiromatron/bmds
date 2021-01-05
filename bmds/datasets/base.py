from typing import Optional

import numpy as np
from pydantic import BaseModel

from ..constants import Dtype


class DatasetBase(BaseModel):
    # Abstract parent-class for dataset-types.

    dtype: Dtype

    class Config:
        extra = "allow"

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

    _dose_linspace: Optional[np.ndarray]

    def dict(self, **kw):
        kw.update(exclude={"_dose_linspace"})
        d = super().dict(**kw)
        d.update(self.kwargs)
        return d

    def to_dict(self):
        # alias used for bmds2
        return self.dict()

    @property
    def dose_linspace(self) -> np.ndarray:
        if not hasattr(self, "_dose_linspace"):
            self._dose_linspace = np.linspace(np.min(self.doses), np.max(self.doses), 100)
        return self._dose_linspace

    def _get_dose_units_text(self):
        return " ({})".format(self.kwargs["dose_units"]) if "dose_units" in self.kwargs else ""

    def _get_response_units_text(self):
        return (
            " ({})".format(self.kwargs["response_units"]) if "response_units" in self.kwargs else ""
        )

    def _get_dataset_name(self):
        return self.kwargs.get("dataset_name", "BMDS output results")
