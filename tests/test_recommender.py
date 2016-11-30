# -*- coding: utf-8 -*-
import textwrap

import bmds
import pytest

from .fixtures import *  # noqa


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
    ✓ BMD exists [bin=✕]
    ✓ BMDL exists [bin=✕]
    ✕ BMDU exists [bin=✓]
    ✓ AIC exists [bin=✕]
    ✓ Residual of interest exists [bin=?]
    ✕ Variance type [bin=?, threshold=0.1]
    ✕ Variance fit [bin=?, threshold=0.1]
    ✓ GGOF [bin=?, threshold=0.1]
    ✓ BMD/BMDL (warning) [bin=✓, threshold=5.0]
    ✓ BMD to BMDL ratio [bin=?, threshold=20.0]
    ✓ Residual of interest [bin=✓, threshold=2.0]
    ✓ Warnings [bin=✓]
    ✓ High BMD [bin=✓, threshold=1.0]
    ✓ High BMDL [bin=?, threshold=1.0]
    ✓ Low BMD (warning) [bin=✓, threshold=3.0]
    ✓ Low BMDL (warning) [bin=✓, threshold=3.0]
    ✓ Low BMD [bin=?, threshold=10.0]
    ✓ Low BMDL [bin=?, threshold=10.0]
    ✕ Control residual [bin=?, threshold=2.0]
    ✕ Control stdev [bin=?, threshold=1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.DICHOTOMOUS).show_rules()

    # dichotomous-cancer:
    txt = dedentify(u"""
    ✓ BMD exists [bin=✕]
    ✓ BMDL exists [bin=✕]
    ✓ BMDU exists [bin=✓]
    ✓ AIC exists [bin=✕]
    ✓ Residual of interest exists [bin=?]
    ✕ Variance type [bin=?, threshold=0.1]
    ✕ Variance fit [bin=?, threshold=0.1]
    ✓ GGOF [bin=?, threshold=0.05]
    ✓ BMD/BMDL (warning) [bin=✓, threshold=5.0]
    ✓ BMD to BMDL ratio [bin=?, threshold=20.0]
    ✓ Residual of interest [bin=✓, threshold=2.0]
    ✓ Warnings [bin=✓]
    ✓ High BMD [bin=✓, threshold=1.0]
    ✓ High BMDL [bin=?, threshold=1.0]
    ✓ Low BMD (warning) [bin=✓, threshold=3.0]
    ✓ Low BMDL (warning) [bin=✓, threshold=3.0]
    ✓ Low BMD [bin=?, threshold=10.0]
    ✓ Low BMDL [bin=?, threshold=10.0]
    ✕ Control residual [bin=?, threshold=2.0]
    ✕ Control stdev [bin=?, threshold=1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.DICHOTOMOUS_CANCER).show_rules()

    # continuous:
    txt = dedentify(u"""
    ✓ BMD exists [bin=✕]
    ✓ BMDL exists [bin=✕]
    ✕ BMDU exists [bin=✓]
    ✓ AIC exists [bin=✕]
    ✓ Residual of interest exists [bin=?]
    ✓ Variance type [bin=?, threshold=0.1]
    ✓ Variance fit [bin=?, threshold=0.1]
    ✓ GGOF [bin=?, threshold=0.1]
    ✓ BMD/BMDL (warning) [bin=✓, threshold=5.0]
    ✓ BMD to BMDL ratio [bin=?, threshold=20.0]
    ✓ Residual of interest [bin=✓, threshold=2.0]
    ✓ Warnings [bin=✓]
    ✓ High BMD [bin=✓, threshold=1.0]
    ✓ High BMDL [bin=?, threshold=1.0]
    ✓ Low BMD (warning) [bin=✓, threshold=3.0]
    ✓ Low BMDL (warning) [bin=✓, threshold=3.0]
    ✓ Low BMD [bin=?, threshold=10.0]
    ✓ Low BMDL [bin=?, threshold=10.0]
    ✓ Control residual [bin=?, threshold=2.0]
    ✓ Control stdev [bin=?, threshold=1.5]
    """)
    assert txt == bmds.Recommender(bmds.constants.CONTINUOUS).show_rules()


def test_apply_logic(cdataset):
    session = bmds.BMDS_v2601(bmds.constants.CONTINUOUS, dataset=cdataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Power)
        session.add_model(bmds.constants.M_Polynomial)
    session.execute()
    session.add_recommender()
    recommended = session.recommend()
    assert recommended is None
    assert session._models[0].logic_bin == 1
    assert len(session._models[0].logic_notes[session._models[0].logic_bin]) == 1
    assert session._models[1].logic_bin == 1
    assert len(session._models[1].logic_notes[session._models[1].logic_bin]) == 1
