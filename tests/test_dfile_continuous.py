import pytest

import bmds


@pytest.fixture
def dataset():
    return bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        responses=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159])


def test_Polynomial_216(dataset):
    model = bmds.models.Polynomial_216(dataset)
    dfile = model.as_dfile()
    expected = 'Polynomial\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n2\n1 5 0\n250 1e-08 1e-08 0 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Polynomial_217(dataset):
    model = bmds.models.Polynomial_217(dataset)
    dfile = model.as_dfile()
    expected = ''
    expected = 'Polynomial\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n2\n1 5 0\n500 1e-08 1e-08 0 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Linear_216(dataset):
    model = bmds.models.Linear_216(dataset)
    dfile = model.as_dfile()
    expected = 'Linear\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1\n1 5 0\n250 1e-08 1e-08 0 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999\n0\n-9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Linear_217(dataset):
    model = bmds.models.Linear_217(dataset)
    dfile = model.as_dfile()
    expected = 'Linear\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1\n1 5 0\n500 1e-08 1e-08 0 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999\n0\n-9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    print(dfile)
    assert dfile == expected


def test_Exponential_M2_17(dataset):
    model = bmds.models.Exponential_M2_17(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M2\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 1000 11 0 1\n250 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M2_19(dataset):
    model = bmds.models.Exponential_M2_19(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M2\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 1000 11 0 1\n500 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M3_17(dataset):
    model = bmds.models.Exponential_M3_17(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M3\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0100 22 0 1\n250 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M3_19(dataset):
    model = bmds.models.Exponential_M3_19(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M3\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0100 22 0 1\n500 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M4_17(dataset):
    model = bmds.models.Exponential_M4_17(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M4\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0010 33 0 1\n250 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M4_19(dataset):
    model = bmds.models.Exponential_M4_19(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M4\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0010 33 0 1\n500 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M5_17(dataset):
    model = bmds.models.Exponential_M5_17(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M5\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0001 44 0 1\n250 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Exponential_M5_19(dataset):
    model = bmds.models.Exponential_M5_19(dataset)
    dfile = model.as_dfile()
    expected = 'Exponential-M5\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0 0001 44 0 1\n500 1e-08 1e-08 0 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Power_216(dataset):
    model = bmds.models.Power_216(dataset)
    dfile = model.as_dfile()
    expected = 'Power\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0\n250 1e-08 1e-08 0 1 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Power_217(dataset):
    model = bmds.models.Power_217(dataset)
    dfile = model.as_dfile()
    expected = 'Power\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0\n500 1e-08 1e-08 0 1 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Hill_216(dataset):
    model = bmds.models.Hill_216(dataset)
    dfile = model.as_dfile()
    expected = 'Hill\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0\n250 1e-08 1e-08 0 1 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected


def test_Hill_217(dataset):
    model = bmds.models.Hill_217(dataset)
    dfile = model.as_dfile()
    expected = 'Hill\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n1 5 0\n500 1e-08 1e-08 0 1 1 0 0\n1 1.0 1 0.95\n-9999 0 -9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999 -9999 -9999\nDose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    assert dfile == expected
