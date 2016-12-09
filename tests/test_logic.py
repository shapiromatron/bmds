# -*- coding: utf-8 -*-
import textwrap

import bmds
from bmds.logic import rules

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
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Power)
        session.add_model(bmds.constants.M_Polynomial)
    session.execute()
    session.add_recommender()
    recommended = session.recommend()
    assert recommended is None
    assert session.models[0].logic_bin == 1
    assert len(session.models[0].logic_notes[session.models[0].logic_bin]) == 1
    assert session.models[1].logic_bin == 1
    assert len(session.models[1].logic_notes[session.models[1].logic_bin]) == 1


def test_exists_rules(cdataset):
    rule = rules.BmdExists(bmds.constants.BIN_FAILURE)

    bin, msg = rule.apply_rule(cdataset, {'BMD': 1})
    assert bin == bmds.constants.BIN_NO_CHANGE
    assert msg is None

    outputs = [
        {},
        {'BMD': -999}
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE
        assert msg == rule.get_failure_message()


def test_greater_than(cdataset):
    rule = rules.GlobalFit(bmds.constants.BIN_FAILURE, threshold=1)

    outputs = [
        {'p_value4': 1.01},
        {'p_value4': 1},
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [
        {'p_value4': 0.99}
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE


def test_less_than(cdataset):
    rule = rules.GlobalFit(bmds.constants.BIN_FAILURE, threshold=1)

    outputs = [
        {'p_value4': 1.01},
        {'p_value4': 1},
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [
        {'p_value4': 0.99}
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE


def test_warnings(cdataset):
    rule = rules.Warnings(bmds.constants.BIN_FAILURE)

    outputs = [
        {},
        {'warnings': []},
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [
        {'warnings': ['failure']}
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE


def test_variance_model(cdataset):
    rule = rules.CorrectVarianceModel(bmds.constants.BIN_FAILURE)

    outputs = [
        {
            'parameters': {'rho': {'value': 1}},
            'p_value2': 0.05
        },
        {
            'parameters': {'rho': {'value': 1}},
            'p_value2': '<0.0001'
        },
        {
            'parameters': {},
            'p_value2': 0.15
        },
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [
        {
            'parameters': {},
            'p_value2': 0.05
        },
        {
            'parameters': {},
            'p_value2': '<0.0001'
        },
        {
            'parameters': {'rho': {'value': 1}},
            'p_value2': 0.15
        },
        {
            'parameters': {'rho': {'value': 1}},
            'p_value2': 'NaN'
        },
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE
