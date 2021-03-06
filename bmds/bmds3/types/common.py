from typing import Any, List

import numpy as np

from ..constants import BMDS_BLANK_VALUE

NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


def list_t_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)


def residual_of_interest(bmd: float, doses: List[float], residuals: List[float]) -> float:
    if bmd <= 0:
        return BMDS_BLANK_VALUE
    diffs = [abs(bmd - dose) for dose in doses]
    index = diffs.index(min(diffs))
    return residuals[index]


class NumpyFloatArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            return np.asarray(v, dtype="float")
        except TypeError:
            raise ValueError("invalid np.ndarray format")
