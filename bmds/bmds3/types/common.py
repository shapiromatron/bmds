from typing import Any, Dict, List

import numpy as np

from ..constants import BMDS_BLANK_VALUE


def list_t_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)


def residual_of_interest(bmd: float, doses: List[float], residuals: List[float]) -> float:
    if bmd <= 0:
        return BMDS_BLANK_VALUE
    diffs = [abs(bmd - dose) for dose in doses]
    index = diffs.index(min(diffs))
    return residuals[index]


def clean_array(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        arr, nan=BMDS_BLANK_VALUE, posinf=BMDS_BLANK_VALUE, neginf=BMDS_BLANK_VALUE
    )


class PydanticNumpyArray(np.ndarray):
    # pydantic friendly numpy arrays

    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def listify(cls, dict_: Dict):
        # convert numpy arrays to lists which can be json serialized
        for key, value in dict_.items():
            if isinstance(value, np.ndarray):
                dict_[key] = value.tolist()
        return dict_


class NumpyIntArray(PydanticNumpyArray):
    @classmethod
    def validate(cls, v):
        try:
            return np.asarray(v, dtype="int")
        except TypeError:
            raise ValueError("invalid np.ndarray format")


class NumpyFloatArray(PydanticNumpyArray):
    # Numpy arrays, agumented
    @classmethod
    def validate(cls, v):
        try:
            return np.asarray(v, dtype="float")
        except TypeError:
            raise ValueError("invalid np.ndarray format")
