import json
import os

import pytest

import bmds
from bmds.bmds3.models import dichotomous
from bmds.bmds3.types.dichotomous import DichotomousModelSettings

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_models(dichds):
    # compare bmd, bmdl, bmdu, aic values
    for Model, expected, aic in [
        (dichotomous.Logistic, [69.583, 61.194, 77.945], 364.0),
        (dichotomous.LogLogistic, [68.163, 59.795, 75.998], 365.0),
        (dichotomous.Probit, [66.874, 58.335, 75.368], 362.1),
        (dichotomous.LogProbit, [66.138, 58.681, 73.153], 366.3),
        (dichotomous.Gamma, [66.037, 57.639, 73.715], 363.6),
        (dichotomous.QuantalLinear, [17.679, 15.645, 20.062], 425.6),
        (dichotomous.Weibull, [64.242, 55.219, 72.814], 358.0),
        (dichotomous.DichotomousHill, [68.173, 59.795, 75.998], 365.0),
    ]:
        model = Model(dichds)
        result = model.execute()
        bmd_values = [result.bmd, result.bmdl, result.bmdu]
        # for regenerating values
        # print(
        #     f"(dichotomous.{Model.__name__}, {np.round(bmd_values, 3).tolist()}, {round(result.aic, 1)})"
        # )
        assert pytest.approx(expected, abs=0.1) == bmd_values
        assert pytest.approx(aic, abs=3.0) == result.aic


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_multistage(dichds):
    # compare bmd, bmdl, bmdu, aic values
    for degree, expected, aic in [
        (1, [17.680, 15.645, 20.062], 425.6),
        (2, [48.016, 44.136, 51.240], 369.7),
        (3, [63.873, 52.260, 72.126], 358.5),
        # TODO - add back higher degrees
        # (4, [69.583, 61.194, 77.945], 363.957),
        # (5, [63.476, 51.826, 78.311], 356.416),
        # (6, [63.665, 11.083, 82.846], 366.384),
        # (7, [69.583, 61.194, 77.945], 363.957),
        # (8, [64.501, 51.366, 85.041], 366.382),
    ]:
        settings = DichotomousModelSettings(degree=degree)
        model = dichotomous.Multistage(dichds, settings)
        result = model.execute()
        bmd_values = [result.bmd, result.bmdl, result.bmdu]
        # for modifying values
        # print(f"({degree}, {np.round(bmd_values, 3).tolist()}, {round(result.aic, 1)})")
        assert pytest.approx(expected, abs=0.1) == bmd_values
        assert pytest.approx(aic, abs=3.0) == result.aic


@pytest.mark.skipif(not should_run, reason=skip_reason)
def test_bmds3_dichotomous_session(dichds):
    session = bmds.session.Bmds330(dataset=dichds)
    session.add_default_models()
    session.execute()
    d = session.to_dict()
    # ensure json-serializable
    print(json.dumps(d))
