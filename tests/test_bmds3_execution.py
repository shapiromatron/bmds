import os
import platform

import pytest

import bmds

# TODO remove this restriction
should_run = platform.system() == "Windows" and os.environ.get("GITHUB_RUN_ID") is None


@pytest.mark.skipif(not should_run, reason="dlls only exist for Windows")
def test_bmds3_dichotomous_session(ddataset):
    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    assert session.version_tuple[0] == 3
    session.add_default_models()
    session.execute()


@pytest.mark.skipif(not should_run, reason="dlls only exist for Windows")
def test_bmds3_continuous(cdataset):
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    assert session.version_tuple[0] == 3
    session.add_default_models()
    session.execute()
