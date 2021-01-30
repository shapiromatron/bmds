import json
import os

import pytest

import bmds
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
        ...

    def test_poly(self, contds, negative_contds):
        ...


@pytest.mark.skipif(not should_run, reason="TODO - figure out why this one randomly fails")
def test_bmds3_increasing(contds):
    """
    Basic tests to ensure AIC and BMD values are successfully created and stable for all model classes
    """
    # test increasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [52.867, 50.457, 55.501], 3181.6),
        (continuous.ExponentialM5, [28.446, 27.025, 30.004], 3070.6),
        (continuous.Power, [25.85, 24.462, 29.223], 3065.8),
        (continuous.Hill, [30.262, 26.124, 34.602], 3072.8),
        (continuous.Linear, [70.738, 67.061, 74.722], 11896.4),
        (continuous.Polynomial, [70.738, 67.061, 74.722], 11896.4),
    ]:
        result = Model(contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.aic, 1)})"
        # print(res)
        assert pytest.approx(actual, abs=0.1) == bmd_values, Model.__name__
        assert pytest.approx(aic, abs=5.0) == result.aic, Model.__name__


def test_bmds3_decreasing(negative_contds):
    # test decreasing means dataset
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [-9999.0, -9999.0, -9999.0], -9999.0),  # TODO -fix
        (continuous.ExponentialM5, [-9999.0, -9999.0, -9999.0], -9999.0),  # TODO -fixs
        (continuous.Power, [58.134, 54.868, 62.789], 3078.0),
        (continuous.Hill, [59.459, 53.449, 68.02], 3083.5),
        (continuous.Linear, [70.426, 66.825, 74.449], 9590.4),
        (continuous.Polynomial, [70.426, 66.825, 74.449], 9590.4),
    ]:
        result = Model(negative_contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"(continuous.{Model.__name__}, {np.round(actual, 3).tolist()}, {round(result.aic, 1)})"
        # print(res)
        assert pytest.approx(actual, abs=0.1) == bmd_values, Model.__name__
        assert pytest.approx(aic, abs=5.0) == result.aic, Model.__name__


# TODO -fix
# @pytest.mark.skipif(True, reason=skip_reason)
# def test_bmds3_variance(contds):
#     model = continuous.Power(contds, dict(disttype=DistType.normal))
#     result = model.execute()
#     assert pytest.approx(result.bmd, abs=0.1) == 123

#     model = continuous.Power(contds, dict(disttype=DistType.normal_ncv))
#     result = model.execute()
#     assert pytest.approx(result.bmd, abs=0.1) == 123

#     model = continuous.Power(contds, dict(disttype=DistType.log_normal))
#     result = model.execute()
#     assert pytest.approx(result.bmd, abs=0.1) == 123


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_polynomial(contds):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [70.738, 67.061, 74.722], 11896.4),
        (2, [65.872, 65.083, 69.618], -9999.0),
        (3, [59.401, 57.257, 62.877], 11686.3),
        (4, [-9999.0, -9999.0, -9999.0], -9999.0),
        (5, [-9999.0, -9999.0, -9999.0], -9999.0),
        (6, [-9999.0, -9999.0, -9999.0], -9999.0),
        (7, [-9999.0, -9999.0, -9999.0], -9999.0),
        (8, [-9999.0, -9999.0, -9999.0], -9999.0),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        result = continuous.Polynomial(contds, settings).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # res = f"({degree}, {np.round(actual, 3).tolist()}, {round(result.aic, 1)})"
        # print(res)
        assert pytest.approx(actual, abs=0.5) == bmd_values, degree
        assert pytest.approx(aic, abs=5.0) == result.aic, degree


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_session(contds: bmds.ContinuousDataset):
    session = bmds.session.Bmds330(dataset=contds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
