from unittest import mock

import pytest
from pydantic import ValidationError

import bmds
from bmds.bmds3.constants import BMDS_BLANK_VALUE, DistType
from bmds.bmds3.recommender import checks
from bmds.bmds3.recommender.recommender import Recommender, RecommenderSettings, Rule, RuleClass
from bmds.constants import Dtype, LogicBin

from .run3 import RunBmds3


class TestRecommenderSettings:
    def test_default(self):
        # assert the default method actually works
        settings = RecommenderSettings.build_default()
        assert isinstance(settings, RecommenderSettings)

    def test_rule_validation(self):
        # assert that the entire rule list must be present
        settings = RecommenderSettings.build_default()
        settings.rules.pop()
        settings2 = settings.json()
        with pytest.raises(ValidationError) as err:
            RecommenderSettings.parse_raw(settings2)
        assert "Rule list must be complete" in str(err)


class TestRecommender:
    def test_df(self):
        df = Recommender().settings.to_df()
        assert df.shape == (22, 6)


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
class TestSessionRecommender:
    def test_apply_logic_dich(self, ddataset2):
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_model(bmds.constants.M_DichotomousHill)
        session.add_model(bmds.constants.M_Gamma)
        session.execute_and_recommend()

        # get model bins
        assert session.recommender.results.model_bin == [0, 0]

        # model recommended and selection is accurate
        assert session.recommender.results.recommended_model_index == 1
        assert session.recommender.results.recommended_model_variable == "aic"
        assert session.models[1].results.fit.aic < session.models[0].results.fit.aic

    def test_apply_logic_cont(self, cdataset):
        session = bmds.session.Bmds330(dataset=cdataset)
        session.add_model(bmds.constants.M_Hill)
        session.add_model(bmds.constants.M_Power)
        session.execute_and_recommend()

        # get model bins
        assert session.recommender.results.model_bin == [1, 1]

        # model recommended and selection is accurate
        assert session.recommender.results.recommended_model_index is None
        assert session.recommender.results.recommended_model_variable is None


class TestChecks:
    def test_exists_rules(self, ddataset):
        model = mock.MagicMock()
        settings = Rule(rule_class=RuleClass.aic_missing, failure_bin=LogicBin.FAILURE)

        # good values
        for value in [-1, 0, 1]:
            model.results.fit.aic = value
            resp = checks.AicExists.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.NO_CHANGE
            assert resp.message == ""

        # bad values
        for value in [None, BMDS_BLANK_VALUE]:
            model.results.fit.aic = value
            resp = checks.AicExists.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.FAILURE
            assert resp.message == "AIC does not exist"

    def test_bmdl_exists_rules(self, ddataset):
        model = mock.MagicMock()
        settings = Rule(rule_class=RuleClass.bmdl_missing, failure_bin=LogicBin.FAILURE)

        # good values
        for value in [1e-8, 10]:
            model.results.fit.aic = value
            resp = checks.BmdlExists.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.NO_CHANGE
            assert resp.message == ""

        # special bad case for bmdl
        for value in [None, 0, BMDS_BLANK_VALUE]:
            model.results.bmdl = value
            resp = checks.BmdlExists.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.FAILURE

    def test_greater_than_rules(self, ddataset):
        model = mock.MagicMock()
        settings = Rule(rule_class=RuleClass.gof, failure_bin=LogicBin.FAILURE, threshold=0.1)

        # good values
        for value in [BMDS_BLANK_VALUE, None, 0.1, 0.11]:
            model.results.gof.p_value = value
            resp = checks.GoodnessOfFit.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.NO_CHANGE
            assert resp.message == ""

        # bad values
        for value in [0.09]:
            model.results.gof.p_value = value
            resp = checks.GoodnessOfFit.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.FAILURE
            assert resp.message == "Goodness of fit p-value less than 0.1"

    def test_less_than_rules(self, ddataset):
        dataset = mock.MagicMock()
        dataset.dtype = Dtype.DICHOTOMOUS
        model = mock.MagicMock()
        settings = Rule(rule_class=RuleClass.roi_large, failure_bin=LogicBin.FAILURE, threshold=2)

        # good values
        for value in [-2, 0, 2]:
            model.results.gof.roi = value
            resp = checks.LargeRoi.check(dataset, model, settings)
            assert resp.logic_bin == LogicBin.NO_CHANGE
            assert resp.message == ""

        # bad values
        for value in [-2.1, 2.1]:
            model.results.gof.roi = value
            resp = checks.LargeRoi.check(dataset, model, settings)
            assert resp.logic_bin == LogicBin.FAILURE
            assert resp.message == "Abs(Residual of interest) greater than 2.0"

    def test_variance_type(self, cdataset):
        model = mock.MagicMock()
        settings = Rule(
            rule_class=RuleClass.variance_type, failure_bin=LogicBin.FAILURE, threshold=0.1
        )
        for disttype, pvalue2, bin in [
            # if pvalue < 0.1 and disttype is normal, fail
            (DistType.normal, 0.09, LogicBin.FAILURE),
            (DistType.log_normal, 0.09, LogicBin.FAILURE),
            (DistType.normal, 0.11, LogicBin.NO_CHANGE),
            (DistType.log_normal, 0.11, LogicBin.NO_CHANGE),
            # if pvalue < 0.1 and disttype is nonconstant, pass
            (DistType.normal_ncv, 0.09, LogicBin.NO_CHANGE),
            (DistType.normal_ncv, 0.11, LogicBin.FAILURE),
        ]:
            model.settings.disttype = disttype
            model.results.tests.p_values = [None, pvalue2]
            resp = checks.VarianceType.check(cdataset, model, settings)
            assert resp.logic_bin == bin

    def test_variance_fit(self, cdataset):
        model = mock.MagicMock()
        settings = Rule(
            rule_class=RuleClass.variance_fit, failure_bin=LogicBin.FAILURE, threshold=0.1
        )
        for disttype, pvalues, bin in [
            # pass if test 2 > 0.1
            (DistType.normal, [1, 0.09, 1, 1], LogicBin.FAILURE),
            (DistType.log_normal, [1, 0.09, 1, 1], LogicBin.FAILURE),
            (DistType.normal, [0, 0.11, 0, 0], LogicBin.NO_CHANGE),
            (DistType.log_normal, [0, 0.11, 0, 0], LogicBin.NO_CHANGE),
            # pass if test 3 > 0.1
            (DistType.normal_ncv, [1, 1, 0.09, 1], LogicBin.FAILURE),
            (DistType.normal_ncv, [0, 0, 0.11, 0], LogicBin.NO_CHANGE),
        ]:
            model.settings.disttype = disttype
            model.results.tests.p_values = pvalues
            resp = checks.VarianceFit.check(cdataset, model, settings)
            assert resp.logic_bin == bin

    def test_zero_df(self, ddataset):
        model = mock.MagicMock()
        settings = Rule(rule_class=RuleClass.dof_zero, failure_bin=LogicBin.FAILURE)

        # good values
        for value in [0.1, 1]:
            model.results.gof.df = value
            resp = checks.NoDegreesOfFreedom.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.NO_CHANGE
            assert resp.message == ""

        # bad values
        for value in [0]:
            model.results.gof.df = value
            resp = checks.NoDegreesOfFreedom.check(ddataset, model, settings)
            assert resp.logic_bin == LogicBin.FAILURE
            assert resp.message == "Zero degrees of freedom; saturated model"
