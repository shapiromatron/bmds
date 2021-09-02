import json

import numpy as np
import pytest
from run3 import RunBmds3

import bmds
from bmds.bmds3.models import dichotomous
from bmds.bmds3.types.dichotomous import DichotomousModelSettings, DichotomousRiskType


class TestBmdModelDichotomous:
    def test_get_param_names(self, ddataset2):
        # test normal model case
        model = dichotomous.Gamma(dataset=ddataset2)
        assert model.get_param_names() == ["g", "a", "b"]

        # test multistage
        model = dichotomous.Multistage(dataset=ddataset2)
        assert model.get_param_names() == ["b0", "b1", "b2"]
        model = dichotomous.Multistage(dataset=ddataset2, settings=dict(degree=3))
        assert model.get_param_names() == ["b0", "b1", "b2", "b3"]

    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_report(self, ddataset2):
        model = dichotomous.Gamma(dataset=ddataset2)
        text = model.text()
        assert "Gamma" in text
        assert "Model has not successfully executed; no results available." in text

        model.execute()
        text = model.text()
        assert "Gamma" in text
        assert "Goodness of fit:" in text

    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_risk_type(self, ddataset2):
        # extra (default)
        model = dichotomous.Logistic(dataset=ddataset2)
        resp1 = model.execute()
        assert model.settings.bmr_type is DichotomousRiskType.ExtraRisk

        # added
        model = dichotomous.Logistic(dataset=ddataset2, settings=dict(bmr_type=0))
        resp2 = model.execute()
        assert model.settings.bmr_type is DichotomousRiskType.AddedRisk

        assert not np.isclose(resp1.bmd, resp2.bmd)
        assert resp1.bmd < resp2.bmd


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_dichotomous_models(ddataset2):
    # compare bmd, bmdl, bmdu, aic values
    for Model, bmd_values, aic in [
        (dichotomous.Logistic, [69.584, 61.193, 77.945], 364.0),
        (dichotomous.LogLogistic, [68.361, 59.795, 76.012], 365.0),
        (dichotomous.Probit, [66.883, 58.333, 75.368], 362.1),
        (dichotomous.LogProbit, [66.149, 58.684, 73.16], 366.3),
        (dichotomous.Gamma, [66.061, 57.639, 73.716], 361.6),
        (dichotomous.QuantalLinear, [17.679, 15.645, 20.062], 425.6),
        (dichotomous.Weibull, [64.26, 55.219, 72.815], 358.4),
        (dichotomous.DichotomousHill, [68.178, 59.795, 75.999], 364.0),
    ]:
        model = Model(ddataset2)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # print(
        #     f"(dichotomous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)})"
        # )
        assert pytest.approx(bmd_values, abs=0.5) == actual
        assert pytest.approx(aic, abs=3.0) == result.fit.aic


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_dichotomous_multistage(ddataset2):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [17.680, 15.645, 20.062], 425.6),
        (2, [48.016, 44.136, 51.240], 369.7),
        (3, [63.873, 52.260, 72.126], 358.5),
        (4, [63.871, 52.073, 72.725], 358.5),
    ]:
        settings = DichotomousModelSettings(degree=degree)
        model = dichotomous.Multistage(ddataset2, settings)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for modifying values
        # print(f"({degree}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)})")
        assert pytest.approx(bmd_values, abs=0.5) == actual
        assert pytest.approx(aic, abs=5.0) == result.fit.aic


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_dichotomous_session(ddataset2):
    session = bmds.session.Bmds330(dataset=ddataset2)
    session.add_default_models()
    session.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
