from typing import NamedTuple

import numpy as np


class ReporterStyleGuide(NamedTuple):
    table: str
    tbl_header: str
    tbl_body: str
    tbl_footnote: str
    outfile: str
    header_1: str
    header_2: str


def float_formatter(value):
    if isinstance(value, str):
        return value
    elif value != 0 and abs(value) < 0.001 or abs(value) > 1e6:
        return "{:.1E}".format(value)
    elif np.isclose(value, int(value)):
        return str(int(value))
    else:
        return "{:.3f}".format(value).rstrip("0")
