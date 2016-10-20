import bmds


def test_logic():
    x = bmds.Session('C', [])
    assert x.execute() == 1
