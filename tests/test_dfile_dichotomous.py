import pytest

import bmds


@pytest.fixture
def dataset():
    return bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14])


def test_Logistic_213(dataset):
    model = bmds.models.Logistic_213(dataset)
    dfile = model.as_dfile()
    expected = 'Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Logistic_214(dataset):
    model = bmds.models.Logistic_214(dataset)
    dfile = model.as_dfile()
    expected = 'Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_LogLogistic_213(dataset):
    model = bmds.models.LogLogistic_213(dataset)
    dfile = model.as_dfile()
    expected = 'LogLogistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_LogLogistic_214(dataset):
    model = bmds.models.LogLogistic_214(dataset)
    dfile = model.as_dfile()
    expected = 'LogLogistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Gamma_215(dataset):
    model = bmds.models.Gamma_215(dataset)
    dfile = model.as_dfile()
    expected = 'Gamma\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Gamma_216(dataset):
    model = bmds.models.Gamma_216(dataset)
    dfile = model.as_dfile()
    expected = 'Gamma\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Probit_32(dataset):
    model = bmds.models.Probit_32(dataset)
    dfile = model.as_dfile()
    expected = 'Probit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Probit_33(dataset):
    model = bmds.models.Probit_33(dataset)
    dfile = model.as_dfile()
    expected = 'Probit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Multistage_32(dataset):
    model = bmds.models.Multistage_32(dataset)
    dfile = model.as_dfile()
    expected = 'Multistage\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Multistage_33(dataset):
    model = bmds.models.Multistage_33(dataset)
    dfile = model.as_dfile()
    expected = 'Multistage\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_MultistageCancer_19(dataset):
    model = bmds.models.MultistageCancer_19(dataset)
    dfile = model.as_dfile()
    expected = 'Multistage-Cancer\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_MultistageCancer_110(dataset):
    model = bmds.models.MultistageCancer_110(dataset)
    dfile = model.as_dfile()
    expected = 'Multistage-Cancer\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Weibull_215(dataset):
    model = bmds.models.Weibull_215(dataset)
    dfile = model.as_dfile()
    expected = 'Weibull\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_Weibull_216(dataset):
    model = bmds.models.Weibull_216(dataset)
    dfile = model.as_dfile()
    expected = 'Weibull\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_LogProbit_32(dataset):
    model = bmds.models.LogProbit_32(dataset)
    dfile = model.as_dfile()
    expected = 'LogProbit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected


def test_LogProbit_33(dataset):
    model = bmds.models.LogProbit_33(dataset)
    dfile = model.as_dfile()
    expected = 'LogProbit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    assert dfile == expected
