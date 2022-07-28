import pytest

from bmds.bmds3.models.base import BmdsLibraryManager

from .run3 import RunBmds3


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
def test_dll_loader():
    BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
