import pytest

import bmds


def test_logistic_213():
    dataset = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14])
    model = bmds.Logistic_213(dataset)
    dfile = model.as_dfile()
    expected = 'Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'
    assert dfile == expected


def test_logistic_214():
    dataset = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14])
    model = bmds.Logistic_214(dataset)
    dfile = model.as_dfile()
    expected = 'Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'
    assert dfile == expected
