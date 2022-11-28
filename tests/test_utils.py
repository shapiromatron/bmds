import bmds
from bmds.utils import get_latest_dll_version


def test_citation():
    citations = bmds.citation()
    assert citations["paper"].startswith("Pham LL")
    assert citations["software"].startswith("Python BMDS.")


def test_get_latest_dll_version():
    version = get_latest_dll_version()
    assert int(version.split(".")[0]) >= 2021  # assume dll in format "YYYY.MM..."
