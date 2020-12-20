import pytest

import bmds
from bmds.bmds3.logic import Recommender, constants


def test_init():
    # Check recommender system can be created for valid data-types.

    # valid dtypes; no errors should be raised
    for dtype in bmds.constants.DTYPES:
        Recommender(dtype=dtype)

    # invalid dtype; error should be raised
    with pytest.raises(ValueError):
        Recommender(dtype="💩")


def test_rules_df():
    # assert dataframe with appropriate shape is created
    df = Recommender(dtype=bmds.constants.DICHOTOMOUS).rules_df()
    assert df.shape == (17, 6)


def test_apply_logic(ddataset):
    session = bmds.session.BMDS_v330(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    session.add_default_models()
    session.execute()
    session.add_recommender()
    recommended = session.recommend()
    assert recommended is not None
    # TODO check logic bins on models


def test_exists_rules(ddataset):
    rule_type = "bmd_missing"
    rule_args = constants.DEFAULT_RULE_ARGS[rule_type]
    Rule = constants.RULE_TYPES[rule_type]
    rule = Rule(**rule_args)

    bin, msg = rule.apply_rule(ddataset, {"bmd": 1})
    assert bin == bmds.constants.BIN_NO_CHANGE
    assert msg is None

    outputs = [{}, {"bmd": -999}]
    for output in outputs:
        bin, msg = rule.apply_rule(ddataset, output)
        assert bin == bmds.constants.BIN_FAILURE
        assert msg == rule.get_failure_message()


def test_greater_than_rules(ddataset):
    rule_type = "gof"
    rule_args = constants.DEFAULT_RULE_ARGS[rule_type]
    rule_args["failure_bin"] = bmds.constants.BIN_FAILURE
    Rule = constants.RULE_TYPES[rule_type]
    rule = Rule(**rule_args)

    threshold = rule_args["threshold"]

    outputs = [{"gof": {"p_value": threshold}}, {"gof": {"p_value": threshold + 0.01}}]
    for output in outputs:
        bin, msg = rule.apply_rule(ddataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [{"gof": {"p_value": threshold - 0.01}}]
    for output in outputs:
        bin, msg = rule.apply_rule(ddataset, output)
        assert bin == bmds.constants.BIN_FAILURE


def test_less_than_rules(ddataset):
    rule_type = "roi_large"
    rule_args = constants.DEFAULT_RULE_ARGS[rule_type]
    rule_args["failure_bin"] = bmds.constants.BIN_FAILURE
    Rule = constants.RULE_TYPES[rule_type]
    rule = Rule(**rule_args)

    threshold = rule_args["threshold"]

    outputs = [{"roi": threshold}, {"roi": threshold - 0.01}]
    for output in outputs:
        bin, msg = rule.apply_rule(ddataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [{"roi": threshold + 0.01}]
    for output in outputs:
        bin, msg = rule.apply_rule(ddataset, output)
        assert bin == bmds.constants.BIN_FAILURE
