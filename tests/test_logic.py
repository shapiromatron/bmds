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
        ({'warnings': ['model has not converged']},
         'Warning(s): model has not converged'),
        ({'warnings': ['model has not converged', 'this happened too']},
         'Warning(s): model has not converged; this happened too'),
    ]
    for output, expected in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE
        assert msg == expected


def test_variance_model(cdataset):
    rule = rules.CorrectVarianceModel(bmds.constants.BIN_FAILURE)

    outputs = [
        {'parameters': {'rho': {'value': 1}}, 'p_value2': 0.05},
        {'parameters': {'rho': {'value': 1}}, 'p_value2': '<0.0001'},
        {'parameters': {}, 'p_value2': 0.15},
    ]
    for output in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [
        (
            {'parameters': {}, 'p_value2': 0.05},
            'Incorrect variance model (p-value 2 = 0.05), constant variance selected'
        ),
        (
            {'parameters': {}, 'p_value2': '<0.0001'},
            'Incorrect variance model (p-value 2 = 0.0001), constant variance selected'
        ),
        (
            {'parameters': {'rho': {'value': 1}}, 'p_value2': 0.15},
            'Incorrect variance model (p-value 2 = 0.15), modeled variance selected'
        ),
        (
            {'parameters': {'rho': {'value': 1}}, 'p_value2': 'NaN'},
            'Correct variance model cannot be determined (p-value 2 = NaN)'
        ),
    ]
    for output, expected in outputs:
        bin, msg = rule.apply_rule(cdataset, output)
        assert bin == bmds.constants.BIN_FAILURE
        assert msg == expected


def test_error_messages(cdataset):
    # Check that error messages are what want them to be.
    # NOTE - Warnings and CorrectVarianceModel rules checked their special tests

    # check existence fields
    rule_classes = [
        (rules.BmdExists, 'BMD does not exist'),
        (rules.BmdlExists, 'BMDL does not exist'),
        (rules.BmduExists, 'BMDU does not exist'),
        (rules.AicExists, 'AIC does not exist'),
        (rules.RoiExists, 'Residual of Interest does not exist'),
    ]
    for rule_class, expected in rule_classes:
        rule = rule_class(bmds.constants.BIN_FAILURE)
        _, msg = rule.apply_rule(cdataset, {})
        assert msg == expected

    # check greater than fields
    rule_classes = [
        (rules.VarianceFit, {'p_value3': 0.09},
         'Variance model fit p-value is less than threshold (0.09 < 0.1)'),
        (rules.GlobalFit, {'p_value4': 0.09},
         'Goodness of fit p-value is less than threshold (0.09 < 0.1)'),
    ]
    for rule_class, outputs, expected in rule_classes:
        rule = rule_class(bmds.constants.BIN_FAILURE, threshold=0.1)
        _, msg = rule.apply_rule(cdataset, outputs)
        assert msg == expected

    # check less-than fields
    max_dose = max(cdataset.doses)
    min_dose = min([dose for dose in cdataset.doses if dose > 0])
    rule_classes = [
        (rules.BmdBmdlRatio, {'BMD': 10, 'BMDL': 1},
         'BMD/BMDL ratio is greater than threshold (10.0 > 1)'),
        (rules.RoiFit, {'residual_of_interest': 2},
         'Residual of interest is greater than threshold (2.0 > 1)'),
        (rules.HighBmd, {'BMD': max_dose + 1},
         'BMD/high dose ratio is greater than threshold (1.0 > 1)'),
        (rules.HighBmdl, {'BMDL': max_dose + 1},
         'BMDL/high dose ratio is greater than threshold (1.0 > 1)'),
        (rules.LowBmd, {'BMD': min_dose - 1},
         'BMD/minimum dose ratio is greater than threshold (1.11 > 1)'),
        (rules.LowBmdl, {'BMDL': min_dose - 1},
         'BMDL/minimum dose ratio is greater than threshold (1.11 > 1)'),
        (rules.ControlResidual, {'fit_residuals': [2]},
         'Residual at lowest dose is greater than threshold (2.0 > 1)'),
        (rules.ControlStdevResiduals, {'fit_est_stdev': [2], 'fit_stdev': [1]},
         'Ratio of modeled to actual stdev. at control is greater than threshold (2.0 > 1)'),
    ]
    for rule_class, outputs, expected in rule_classes:
        rule = rule_class(bmds.constants.BIN_FAILURE, threshold=1)
        _, msg = rule.apply_rule(cdataset, outputs)
        assert msg == expected