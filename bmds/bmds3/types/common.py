from typing import Any, List

BMDS_BLANK_VALUE = -9999
NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


def list_t_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)


def residual_of_interest(bmd: float, doses: List[float], residuals: List[float]) -> float:
    if bmd > 0 and len(doses) > 0:
        diff = abs(doses[0] - bmd)
        r = residuals[0]
        for i, val in enumerate(doses):
            if abs(val - bmd) < diff:
                diff = abs(val - bmd)
                r = residuals[i]
        return r
    return BMDS_BLANK_VALUE
