import pytest

from bmds.bmds3.models.base import BmdsLibraryManager

from ..run3 import RunBmds3


class TestBmdsLibraryManager:
    @pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
    def test_330(self):
        BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
