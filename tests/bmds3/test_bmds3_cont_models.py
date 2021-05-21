import json
import os

# import numpy as np
import pytest

import bmds
from bmds.bmds3.constants import DistType
from bmds.bmds3.models import continuous
from bmds.bmds3.types.continuous import ContinuousModelSettings

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def contds():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[10, 20, 30, 40, 50],
        stdevs=[3, 4, 5, 6, 7],
    )


@pytest.fixture
def negative_contds():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[1, -5, -10, -20, -30],
        stdevs=[3, 4, 5, 6, 7],
    )


class TestPriorOverrides:
    def test_exp5(self, contds, negative_contds):
        model = continuous.ExponentialM5(contds)
        model.settings.priors.priors[2].name == "c"
        assert model.settings.priors.priors[2].min_value == 0
        assert model.settings.priors.priors[2].max_value == 18

        model = continuous.ExponentialM5(negative_contds)
        model.settings.priors.priors[2].name == "c"
        assert model.settings.priors.priors[2].min_value == -18
        assert model.settings.priors.priors[2].max_value == 0

    def test_hill(self, contds, negative_contds):
        # TODO - add ...
        ...

    def test_poly(self, contds, negative_contds):
        # TODO - add ...
        ...


class TestBmdModelContinuous:
    def test_get_param_names(self, contds):
        # test normal model case
        for m in [
            continuous.Power(dataset=contds),
            continuous.Power(dataset=contds, settings=dict(disttype=DistType.normal)),
            continuous.Power(dataset=contds, settings=dict(disttype=DistType.log_normal)),
        ]:
            assert m.get_param_names() == ["g", "v", "n", "rho"]
        m = continuous.Power(dataset=contds, settings=dict(disttype=DistType.normal_ncv))
        assert m.get_param_names() == ["g", "v", "n", "rho", "alpha"]

        # test polynomial
        model = continuous.Linear(dataset=contds)
        assert model.get_param_names() == ["b0", "b1", "rho"]
        model = continuous.Polynomial(dataset=contds)
        assert model.get_param_names() == ["b0", "b1", "b2", "rho"]
        model = continuous.Polynomial(dataset=contds, settings=dict(degree=3))
        assert model.get_param_names() == ["b0", "b1", "b2", "b3", "rho"]
        model = continuous.Polynomial(
            dataset=contds, settings=dict(degree=3, disttype=DistType.normal_ncv)
        )
        assert model.get_param_names() == ["b0", "b1", "b2", "b3", "rho", "alpha"]

    @pytest.mark.skipif(not should_run, reason=skip_reason)
    def test_report(self, contds):
        model = continuous.Hill(dataset=contds)
        text = model.report()
        assert "Hill" in text
        assert "Execution was not completed." in text

        model.execute()
        text = model.report()
        assert "Hill" in text
        assert "Analysis of Deviance" in text


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_increasing(contds):
    """
    Basic tests to ensure AIC and BMD values are successfully created and stable for all model classes
    """
    # test increasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [157.492, 154.493, 169.93], 3842.8),
        # (continuous.ExponentialM5, [nan, nan, nan], 3069.8),
        (continuous.Power, [25.852, 24.4, 29.599], 3067.8),
        (continuous.Hill, [29.25, 25.893, 33.209], 3071.3),
        (continuous.Linear, [25.856, 24.388, 27.451], 3065.8),
        (continuous.Polynomial, [25.55, 23.836, 27.57], 3068.0),
    ]:
        result = Model(contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_decreasing(negative_contds):
    # test decreasing means dataset
    for Model, bmd_values, aic in [
        # (continuous.ExponentialM3, [nan, nan, nan], 4296.0),
        # (continuous.ExponentialM5, [nan, nan, nan], 4296.3),
        (continuous.Power, [56.5, 54.3, 59.7], 3077.5),
        # (continuous.Hill, [nan, nan, nan], 3925.8),
        (continuous.Linear, [35.3, 33.1, 37.7], 3115.3),
        (continuous.Polynomial, [51.5, 47.3, 56.5], 3074.7),
    ]:
        model = Model(negative_contds)
        result = model.execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 1).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(bmd_values, rel=0.05) == actual, Model.__name__
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, Model.__name__


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_variance(contds):
    model = continuous.Power(contds, dict(disttype=DistType.normal))
    result = model.execute()
    assert model.settings.disttype is DistType.normal
    assert pytest.approx(result.bmd, abs=1.0) == 25.85
    assert len(result.parameters.values) == 4

    model = continuous.Power(contds, dict(disttype=DistType.normal_ncv))
    result = model.execute()
    assert model.settings.disttype is DistType.normal_ncv
    assert len(result.parameters.values) == 5
    assert pytest.approx(result.bmd, abs=1.0) == 13.3

    # TODO -fix - currently segfault
    # model = continuous.Power(contds, dict(disttype=DistType.log_normal))
    # result = model.execute()
    # assert model.settings.disttype is DistType.log_normal
    # assert pytest.approx(result.bmd, abs=0.1) == 123
    # assert len(result.parameters.values) == 4


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_polynomial(contds):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [25.856, 24.388, 27.451], 3065.8),
        (2, [25.55, 23.836, 27.57], 3068.0),
        (3, [25.681, 25.62, 26.083], 3070.1),
        # (4, [-9999.0, -9999.0, -9999.0], -9999.0),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        result = continuous.Polynomial(contds, settings).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"({degree}, {np.round(actual, 3).tolist()}, {round(result.fit.aic, 1)}),"
        # print(res)
        assert pytest.approx(actual, rel=0.05) == bmd_values, degree
        assert pytest.approx(aic, rel=0.01) == result.fit.aic, degree


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_session(contds):
    session = bmds.session.Bmds330(dataset=contds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_individual_session(cidataset):
    session = bmds.session.Bmds330(dataset=cidataset)
    session.add_model(bmds.constants.M_Power)
    session.add_model(bmds.constants.M_Hill)
    session.add_model(bmds.constants.M_ExponentialM3)
    session.add_model(bmds.constants.M_ExponentialM5)
    session.add_model(bmds.constants.M_Linear)
    session.add_model(bmds.constants.M_Polynomial, {"degree": 2})
    # session.add_model(bmds.constants.M_Polynomial, {"degree": 3})  # TODO - segfault
    session.execute()
    for model in session.models:
        model.results = model.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
