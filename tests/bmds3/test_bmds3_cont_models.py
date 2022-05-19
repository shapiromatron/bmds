import json

# import numpy as np
import pytest
from run3 import RunBmds3

import bmds
from bmds import constants
from bmds.bmds3.constants import BMDS_BLANK_VALUE, DistType
from bmds.bmds3.models import continuous
from bmds.bmds3.types.continuous import ContinuousModelSettings


class TestPriorOverrides:
    def test_exp5(self, cdataset2, negative_cdataset):
        model = continuous.ExponentialM5(cdataset2)
        model.settings.priors.priors[2].name == "c"
        assert model.settings.priors.priors[2].min_value == 0
        assert model.settings.priors.priors[2].max_value == 18

        model = continuous.ExponentialM5(negative_cdataset)
        model.settings.priors.priors[2].name == "c"
        assert model.settings.priors.priors[2].min_value == -18
        assert model.settings.priors.priors[2].max_value == 0

    def test_hill(self, cdataset2):
        cv = continuous.Hill(cdataset2, settings=dict(disttype=DistType.normal)).settings.priors
        assert cv.get_prior("v").min_value == -1e8

        ln = continuous.Hill(cdataset2, settings=dict(disttype=DistType.log_normal)).settings.priors
        assert ln.get_prior("v").min_value == -1e8

        ncv = continuous.Hill(
            cdataset2, settings=dict(disttype=DistType.normal_ncv)
        ).settings.priors
        assert ncv.get_prior("v").min_value == -1000

    def test_poly(self, cdataset2, negative_cdataset):
        model = continuous.Polynomial(cdataset2)
        model.settings.priors.priors[1].min_value == 0
        model.settings.priors.priors[1].max_value == 1e8
        model.settings.priors.priors[2].min_value == 0
        model.settings.priors.priors[2].max_value == 1e8

        model = continuous.Polynomial(negative_cdataset)
        model.settings.priors.priors[1].min_value = -1e8
        model.settings.priors.priors[1].max_value == 0
        model.settings.priors.priors[2].min_value == -1e8
        model.settings.priors.priors[2].max_value == 0


class TestBmdModelContinuous:
    def test_get_param_names(self, cdataset2):
        # test normal model case
        for m in [
            continuous.Power(dataset=cdataset2),
            continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.normal)),
            continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.log_normal)),
        ]:
            assert m.get_param_names() == ["g", "v", "n", "rho"]
        m = continuous.Power(dataset=cdataset2, settings=dict(disttype=DistType.normal_ncv))
        assert m.get_param_names() == ["g", "v", "n", "rho", "alpha"]

        # test polynomial
        model = continuous.Linear(dataset=cdataset2)
        assert model.get_param_names() == ["b0", "b1", "rho"]
        model = continuous.Polynomial(dataset=cdataset2)
        assert model.get_param_names() == ["b0", "b1", "b2", "rho"]
        model = continuous.Polynomial(dataset=cdataset2, settings=dict(degree=3))
        assert model.get_param_names() == ["b0", "b1", "b2", "b3", "rho"]
        model = continuous.Polynomial(
            dataset=cdataset2, settings=dict(degree=3, disttype=DistType.normal_ncv)
        )
        assert model.get_param_names() == ["b0", "b1", "b2", "b3", "rho", "alpha"]

    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_report(self, cdataset2):
        model = continuous.Hill(dataset=cdataset2)
        text = model.text()
        assert "Hill" in text
        assert "Model has not successfully executed; no results available." in text

        model.execute()
        text = model.text()
        assert "Hill" in text
        assert "Goodness of fit:" in text


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_increasing(cdataset2):
    """
    Basic tests to ensure AIC and BMD values are successfully created and stable for all model classes
    """
    # test increasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [52.866, 50.493, 55.422], 3187.6),
        (continuous.ExponentialM5, [25.955, 24.578, 27.501], 3071.8),
        (continuous.Power, [25.843, 24.357, 29.769], 3067.8),
        (continuous.Hill, [30.435, 24.451, 34.459], 3074.6),
        (continuous.Linear, [25.851, 24.355, 27.528], 3067.8),
        (continuous.Polynomial, [25.866, 24.351, 28.653], 3067.8),
    ]:
        result = Model(cdataset2).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_decreasing(negative_cdataset):
    # test decreasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [BMDS_BLANK_VALUE, BMDS_BLANK_VALUE, BMDS_BLANK_VALUE], 4296.3),
        (continuous.ExponentialM5, [BMDS_BLANK_VALUE, BMDS_BLANK_VALUE, BMDS_BLANK_VALUE], 4298.3),
        (continuous.Power, [56.5, 49.8, 63.6], 3079.5),
        (continuous.Hill, [57.7, 51.0, 64.9], 3082.5),
        (continuous.Linear, [35.3, 33.1, 37.8], 3117.3),
        (continuous.Polynomial, [52.5, 46.2, 59.9], 3076.6),
    ]:
        model = Model(negative_cdataset)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 1).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_variance(cdataset2):
    model = continuous.Power(cdataset2, dict(disttype=DistType.normal))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.2f}, {actual[1]:.2f}, {actual[2]:.2f}")
    assert model.settings.disttype is DistType.normal
    assert pytest.approx(actual, rel=0.05) == [25.81, 24.32, 29.73]
    assert len(result.parameters.values) == 4

    model = continuous.Power(cdataset2, dict(disttype=DistType.normal_ncv))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.2f}, {actual[1]:.2f}, {actual[2]:.2f}")
    assert model.settings.disttype is DistType.normal_ncv
    assert len(result.parameters.values) == 5
    assert pytest.approx(actual, rel=0.05) == [14.43, 13.03, 14.73]

    # only Power and Exp can be used
    model = continuous.ExponentialM3(cdataset2, dict(disttype=DistType.log_normal))
    result = model.execute()
    actual = [result.bmd, result.bmdl, result.bmdu]
    # print(f"{actual[0]:.2f}, {actual[1]:.2f}, {actual[2]:.2f}")
    assert model.settings.disttype is DistType.log_normal
    assert pytest.approx(actual, rel=0.05) == [104.59, 93.19, 118.99]
    assert len(result.parameters.values) == 5


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_continuous_polynomial(cdataset2):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [25.84, 24.358, 27.529], 3067.8),
        (2, [25.984, 24.328, 28.642], 3069.8),
        (3, [26.803, 24.259, 28.908], 3070.3),
        (4, [26.137, 24.336, 28.713], 3069.9),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        result = continuous.Polynomial(cdataset2, settings).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"({degree}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(actual, rel=0.05) == bmd_values, degree
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, degree


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_bmds3_continuous_session(cdataset2):
    session = bmds.session.Bmds330(dataset=cdataset2)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_increasing_lognormal(cdataset2):
    session = bmds.session.Bmds330(dataset=cdataset2)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_ExponentialM3, constants.M_ExponentialM5):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is True
        assert model.results.bmd != BMDS_BLANK_VALUE

    session = bmds.session.Bmds330(dataset=cdataset2)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_Hill, constants.M_Power, constants.M_Polynomial):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is False
        assert model.results.bmd == BMDS_BLANK_VALUE


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_decreasing_lognormal(negative_cdataset):
    session = bmds.session.Bmds330(dataset=negative_cdataset)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_ExponentialM3, constants.M_ExponentialM5):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is False  # TODO - should return valid value
        assert model.results.bmd == BMDS_BLANK_VALUE

    session = bmds.session.Bmds330(dataset=negative_cdataset)
    settings = dict(disttype=DistType.log_normal)
    for model in (constants.M_Hill, constants.M_Power, constants.M_Polynomial):
        session.add_model(model, settings)
    session.execute()
    for model in session.models:
        assert model.results.has_completed is False
        assert model.results.bmd == BMDS_BLANK_VALUE


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_infinite_bmr_values():
    # check that infinite BMR values are correctly set to BMDS_BLANK_VALUE
    ds = bmds.ContinuousDataset(
        doses=[0.0, 50, 100],
        ns=[20, 20, 20],
        means=[5.26, 5.76, 6.13],
        stdevs=[2.53, 1.47, 2.47],
    )
    model = continuous.ExponentialM3(ds)
    model.execute()
    assert model.results.plotting.bmdu_y == BMDS_BLANK_VALUE
