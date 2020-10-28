import os
import platform

import pytest

import bmds

# TODO remove this restriction
should_run = platform.system() == "Darwin" and os.getenv("CI") is None


@pytest.mark.skipif(not should_run, reason="dlls only exist for Mac")
def test_bmds3_dichotomous_session():
    ds = bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ds)
    assert session.version_tuple == (3, 3, 0)
    session.add_default_models()
    for model in session.models:
        model.execute(debug=True)
    session.to_dict(0)
