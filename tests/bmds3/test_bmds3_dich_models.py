import json
import os

import numpy as np
import pytest

import bmds
from bmds.bmds3.models import dichotomous

# TODO remove this restriction
should_run = os.getenv("CI") is None


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.mark.skipif(not should_run, reason="dlls not present on CI")
def test_bmds3_dichotomous_models(dichds):
    # compare bmd, bmdl, bmdu, aic values
    for Model, expected in [
        (dichotomous.Logistic, [69.583, 61.194, 77.945, 363.957]),
        (dichotomous.LogLogistic, [68.163, 59.795, 75.998, 364.982]),
        (dichotomous.Probit, [66.874, 58.335, 75.368, 362.057]),
        (dichotomous.LogProbit, [66.138, 58.681, 73.153, 366.271]),
        (dichotomous.Gamma, [66.037, 57.639, 73.715, 363.607]),
        (dichotomous.QuantalLinear, [17.679, 15.645, 20.062, 425.594]),
        (dichotomous.Weibull, [64.242, 55.219, 72.814, 362.400]),
        (dichotomous.DichotomousHill, [68.173, 59.795, 75.998, 366.982]),
        (dichotomous.Multistage, [64.127, 52.552, 76.220, 364.384]),
    ]:
        result = Model(dichds).execute()
        actual = [result.bmd, result.bmdl, result.bmdu, result.aic]
        # for regenerating expected: `print(np.round(actual, 3).tolist())`
        assert np.isclose(np.array(actual), np.array(expected), atol=1e-3).all()


@pytest.mark.skipif(not should_run, reason="dlls not present on CI")
def test_bmds3_dichotomous_session(dichds):
    session = bmds.session.BMDS_v330(bmds.constants.DICHOTOMOUS, dataset=dichds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict(0)
    # ensure json-serializable
    print(json.dumps(d))


@pytest.mark.skipif(not should_run, reason="TODO - fix")
def test_bmds3_dichotomous_ma_session(dichds):
    session = bmds.session.BMDS_v330(bmds.constants.DICHOTOMOUS, dataset=dichds)
    # session.add_default_models()
    session.add_model(bmds.constants.M_Logistic)
    session.add_model(bmds.constants.M_LogLogistic)
    session.add_model_averaging()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict(0)
    # ensure json-serializable
    print(json.dumps(d))
