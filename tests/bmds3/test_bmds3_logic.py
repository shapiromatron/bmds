import os

import pytest

import bmds
from bmds.bmds3.logic import Recommender, constants


# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


class TestRecommender:
    def test_init(self):
        # valid dtypes; no errors should be raised
        for dtype in bmds.constants.DTYPES:
            Recommender(dtype=dtype)

        # invalid dtype; error should be raised
        with pytest.raises(ValueError):
            Recommender(dtype="💩")

    def test_df(self):
        # assert dataframe with appropriate shape is created
        df = Recommender(dtype=bmds.constants.DICHOTOMOUS).to_df()
        assert df.shape == (17, 6)


@pytest.mark.skipif(not should_run, reason=skip_reason)
class TestSessionRecommender:
    def test_apply_logic_dich(self, dichds):
        session = bmds.session.Bmds330(dataset=dichds)
        session.add_model(bmds.constants.M_DichotomousHill)
        session.add_model(bmds.constants.M_Gamma)
        session.execute()
        session.add_recommender()
        recommended = session.recommend()

        assert (
            session.models[0].model_name() == "Hill" and session.models[1].model_name() == "Gamma"
        )
        # both models have no bin warnings/errors
        assert session.models[0].logic_bin == 0
        assert session.models[1].logic_bin == 0
        # however, gamma has a smaller AIC
        assert session.models[1].results.aic < session.models[0].results.aic
        assert recommended.model_name() == "Gamma"

    def test_apply_logic_cont(self, cdataset):
        session = bmds.session.Bmds330(dataset=cdataset)
        session.add_model(bmds.constants.M_Hill)
        session.add_model(bmds.constants.M_Power)
        session.execute()
        session.add_recommender()
        recommended = session.recommend()

        assert recommended is None
        assert session.models[0].logic_bin == 1
        assert len(session.models[0].logic_notes[session.models[0].logic_bin]) == 1
        assert session.models[1].logic_bin == 1
        assert len(session.models[1].logic_notes[session.models[1].logic_bin]) == 1


def test_exists_rules(ddataset):
    rule_type = "bmd_missing"
    rule_args = constants.DEFAULT_RULE_ARGS[rule_type]
    Rule = constants.RULE_TYPES[rule_type]
    rule = Rule(**rule_args)

    bin, msg = rule.apply_rule(None, ddataset, {"bmd": 1})
    assert bin == bmds.constants.BIN_NO_CHANGE
    assert msg is None

    outputs = [{}, {"bmd": -999}]
    for output in outputs:
        bin, msg = rule.apply_rule(None, ddataset, output)
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
        bin, msg = rule.apply_rule(None, ddataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [{"gof": {"p_value": threshold - 0.01}}]
    for output in outputs:
        bin, msg = rule.apply_rule(None, ddataset, output)
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
        bin, msg = rule.apply_rule(None, ddataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [{"roi": threshold + 0.01}]
    for output in outputs:
        bin, msg = rule.apply_rule(None, ddataset, output)
        assert bin == bmds.constants.BIN_FAILURE


def test_zero_df(ddataset):
    rule_type = "dof_zero"
    rule_args = constants.DEFAULT_RULE_ARGS[rule_type]
    rule_args["failure_bin"] = bmds.constants.BIN_FAILURE
    Rule = constants.RULE_TYPES[rule_type]
    rule = Rule(**rule_args)

    outputs = [{"gof": {"df": 1}}, {"gof": {"df": 0.01}}]
    for output in outputs:
        bin, msg = rule.apply_rule(None, ddataset, output)
        assert bin == bmds.constants.BIN_NO_CHANGE
        assert msg is None

    outputs = [{"gof": {"df": 0}}]
    for output in outputs:
        bin, msg = rule.apply_rule(None, ddataset, output)
        assert bin == bmds.constants.BIN_FAILURE
