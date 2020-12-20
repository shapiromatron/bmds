import json
import os

import pytest

import bmds

# TODO remove this restriction
should_run = os.getenv("CI") is None


@pytest.mark.skipif(not should_run, reason="dlls not present on CI")
def test_bmds3_dichotomous_session():
    ds = bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )
    session = bmds.session.BMDS_v330(bmds.constants.DICHOTOMOUS, dataset=ds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict(0)
    # ensure json-serializable
    print(json.dumps(d))


@pytest.mark.skipif(not should_run, reason="TODO - fix")
def test_bmds3_dichotomous_ma_session():
    ds = bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )
    session = bmds.session.BMDS_v330(bmds.constants.DICHOTOMOUS, dataset=ds)
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
