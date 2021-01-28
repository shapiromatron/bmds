import json
import os

import pytest

import bmds
from bmds.bmds3.models import continuous
from bmds.bmds3.types.continuous import ContinuousModelSettings, DistType

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


@pytest.mark.skipif(not should_run, reason="TODO - figure out why this one randomly fails")
def test_bmds3_continuous_models(contds):
    # compare bmd, bmdl, bmdu, aic values
    for Model, bmd_values, aic in [
        (continuous.ExponentialM3, [20.220, 19.211, 21.845], 3368.294),
        (continuous.ExponentialM5, [31.270, 29.849, 34.013], 3073.712),
        (continuous.Power, [-9999.0, -9999.0, -9999.0], 3067.833),
        (continuous.Hill, [-9999.0, -9999.0, -9999.0], 3071.148),
        (continuous.Linear, [25.839, 24.384, 27.457], 3065.833),
    ]:
        result = Model(contds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        assert pytest.approx(actual, abs=0.1) == bmd_values, Model.__name__
        assert pytest.approx(aic, abs=5.0) == result.aic, Model.__name__


@pytest.mark.skipif(True, reason=skip_reason)  # TODO -fix
def test_bmds3_variance(contds):
    model = continuous.Power(contds, dict(disttype=DistType.normal))
    result = model.execute()
    assert pytest.approx(result.bmd, abs=0.1) == 123

    model = continuous.Power(contds, dict(disttype=DistType.normal_ncv))
    result = model.execute()
    assert pytest.approx(result.bmd, abs=0.1) == 123

    model = continuous.Power(contds, dict(disttype=DistType.log_normal))
    result = model.execute()
    assert pytest.approx(result.bmd, abs=0.1) == 123


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_polynomial(contds):
    # compare bmd, bmdl, bmdu, aic values
    for degree, bmd_values, aic in [
        (1, [25.839, 24.384, 27.457], 3063.833),
        (2, [25.612, 24.191, 27.126], -9999.0),
        (3, [25.728, 25.364, 26.034], -9999.0),
    ]:
        settings = ContinuousModelSettings(degree=degree)
        result = continuous.Polynomial(contds, settings).execute()
        actual = [result.bmd, result.bmdl, result.bmdu]
        assert pytest.approx(actual, abs=0.5) == bmd_values, degree
        assert pytest.approx(aic, abs=5.0) == result.aic, degree


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_continuous_session(contds: bmds.ContinuousDataset):
    session = bmds.session.Bmds330(dataset=contds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
