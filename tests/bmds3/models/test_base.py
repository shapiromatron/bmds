from bmds.bmds3.models.base import BmdsLibraryManager


class TestBmdsLibraryManager:
    def test_330(self):
        BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
