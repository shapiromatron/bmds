import pytest
from bmds3.run3 import RunBmds3

import bmds
from bmds.utils import get_latest_dll_version


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_citation():
    citations = bmds.citation()
    assert citations["paper"].startswith("Pham LL")
    assert citations["software"].startswith("Python BMDS.")


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_get_latest_dll_version():
    version = get_latest_dll_version()
    assert int(version.split(".")[0]) >= 2021
