from typing import Union

import numpy as np
from pydantic import BaseModel, validator


def string_to_floats(val):
    np.array(val.split(","), dtype=np.float).tolist()
    return val


class DichotomousDatasetSchema(BaseModel):

    id: str

    doses: str
    ns: str
    incidences: str

    _validate_doses = validator("doses", allow_reuse=True)(string_to_floats)
    _validate_ns = validator("ns", allow_reuse=True)(string_to_floats)
    _validate_incidences = validator("incidences", allow_reuse=True)(string_to_floats)


class ContinuousDatasetSchema(BaseModel):

    id: str

    doses: str
    ns: str
    means: str
    stdevs: str

    _validate_doses = validator("doses", allow_reuse=True)(string_to_floats)
    _validate_ns = validator("ns", allow_reuse=True)(string_to_floats)
    _validate_means = validator("means", allow_reuse=True)(string_to_floats)
    _validate_stdevs = validator("stdevs", allow_reuse=True)(string_to_floats)


def nan_to_default(val):
    return -999 if isinstance(val, str) or np.isnan(val) else val


class ResultSchema(BaseModel):
    dataset_id: str
    bmds_version: str
    model: str
    completed: bool
    inputs: dict = {}
    outputs: dict = {}
    bmd: Union[float, str] = -999
    bmdl: Union[float, str] = -999
    bmdu: Union[float, str] = -999
    aic: Union[float, str] = -999

    _validate_bmd = validator("bmd", allow_reuse=True)(nan_to_default)
    _validate_bmdl = validator("bmdl", allow_reuse=True)(nan_to_default)
    _validate_bmdu = validator("bmdu", allow_reuse=True)(nan_to_default)
    _validate_aic = validator("aic", allow_reuse=True)(nan_to_default)
