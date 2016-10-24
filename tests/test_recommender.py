# -*- coding: utf-8 -*-
import textwrap

import bmds
import pytest


def dedentify(txt):
    return textwrap.dedent(txt).strip()


def test_init():
    # Check recommender system can be created for valid data-types.

    # valid dtypes; no errors should be raised
    for dtype in bmds.constants.DTYPES:
        bmds.Recommender(dtype)

    # invalid dtype; error should be raised
    with pytest.raises(ValueError):
        bmds.Recommender(u'💩')


def test_default_logic():
    # Check that default logic is as expected.

    # dichotomous:
    txt = dedentify(u"""
    ✓ BMD exists
    ✓ BMDL exists
    ✕ BMDU exists
    ✓ AIC exists
    ✓ Residual of interest exists
    ✕ Variance type: [threshold = 0.1]
    ✕ Variance fit: [threshold = 0.1]
    ✓ GGOF: [threshold = 0.1]
    ✓ BMD/BMDL (warning): [threshold = 5.0]
    ✓ BMD to BMDL ratio: [threshold = 20.0]
    ✓ Residual of interest: [threshold = 2.0]
    ✓ Warnings
    ✓ High BMD: [threshold = 1.0]
    ✓ High BMDL: [threshold = 1.0]
    ✓ Low BMD (warning): [threshold = 3.0]
    ✓ Low BMDL (warning): [threshold = 3.0]
    ✓ Low BMD: [threshold = 10.0]
    ✓ Low BMDL: [threshold = 10.0]
    ✕ Control residual: [threshold = 2.0]
    ✕ Control stdev: [threshold = 1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.DICHOTOMOUS).show_rules()

    # dichotomous-cancer:
    txt = dedentify(u"""
    ✓ BMD exists
    ✓ BMDL exists
    ✓ BMDU exists
    ✓ AIC exists
    ✓ Residual of interest exists
    ✕ Variance type: [threshold = 0.1]
    ✕ Variance fit: [threshold = 0.1]
    ✓ GGOF: [threshold = 0.05]
    ✓ BMD/BMDL (warning): [threshold = 5.0]
    ✓ BMD to BMDL ratio: [threshold = 20.0]
    ✓ Residual of interest: [threshold = 2.0]
    ✓ Warnings
    ✓ High BMD: [threshold = 1.0]
    ✓ High BMDL: [threshold = 1.0]
    ✓ Low BMD (warning): [threshold = 3.0]
    ✓ Low BMDL (warning): [threshold = 3.0]
    ✓ Low BMD: [threshold = 10.0]
    ✓ Low BMDL: [threshold = 10.0]
    ✕ Control residual: [threshold = 2.0]
    ✕ Control stdev: [threshold = 1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.DICHOTOMOUS_CANCER).show_rules()

    # continuous:
    txt = dedentify(u"""
    ✓ BMD exists
    ✓ BMDL exists
    ✕ BMDU exists
    ✓ AIC exists
    ✓ Residual of interest exists
    ✓ Variance type: [threshold = 0.1]
    ✓ Variance fit: [threshold = 0.1]
    ✓ GGOF: [threshold = 0.1]
    ✓ BMD/BMDL (warning): [threshold = 5.0]
    ✓ BMD to BMDL ratio: [threshold = 20.0]
    ✓ Residual of interest: [threshold = 2.0]
    ✓ Warnings
    ✓ High BMD: [threshold = 1.0]
    ✓ High BMDL: [threshold = 1.0]
    ✓ Low BMD (warning): [threshold = 3.0]
    ✓ Low BMDL (warning): [threshold = 3.0]
    ✓ Low BMD: [threshold = 10.0]
    ✓ Low BMDL: [threshold = 10.0]
    ✓ Control residual: [threshold = 2.0]
    ✓ Control stdev: [threshold = 1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.CONTINUOUS).show_rules()
