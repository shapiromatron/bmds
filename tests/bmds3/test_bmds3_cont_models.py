import json
import os

import pytest

import bmds

# TODO remove this restriction
should_run = os.getenv("CI") is None


@pytest.mark.skipif(not should_run, reason="dlls only exist for Mac")
def test_bmds3_continuous_session():
    ds = bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[10, 20, 30, 40, 50],
        stdevs=[3, 4, 5, 6, 7],
    )
    session = bmds.session.BMDS_v330(bmds.constants.CONTINUOUS, dataset=ds)
    session.add_default_models()
    session.execute()
    for model in session.models:
        model.results = model.execute(debug=True)
    d = session.to_dict(0)
    # ensure json-serializable
    print(json.dumps(d))
