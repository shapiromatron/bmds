from typing import Any, List

BMDS_BLANK_VALUE = -9999
NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


def list_t_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)
