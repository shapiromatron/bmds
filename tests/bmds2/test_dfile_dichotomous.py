from bmds import bmds2


def test_Logistic_215(ddataset):
    model = bmds2.models.Logistic_215(ddataset)
    dfile = model.as_dfile()
    expected = "Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_LogLogistic_215(ddataset):
    model = bmds2.models.LogLogistic_215(ddataset)
    dfile = model.as_dfile()
    expected = "LogLogistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_Gamma_217(ddataset):
    model = bmds2.models.Gamma_217(ddataset)
    dfile = model.as_dfile()
    expected = "Gamma\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_Probit_34(ddataset):
    model = bmds2.models.Probit_34(ddataset)
    dfile = model.as_dfile()
    expected = "Probit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_Multistage_34(ddataset):
    model = bmds2.models.Multistage_34(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_MultistageCancer_34(ddataset):
    model = bmds2.models.MultistageCancer_34(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage-Cancer\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_Weibull_217(ddataset):
    model = bmds2.models.Weibull_217(ddataset)
    dfile = model.as_dfile()
    expected = "Weibull\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_LogProbit_34(ddataset):
    model = bmds2.models.LogProbit_34(ddataset)
    dfile = model.as_dfile()
    expected = "LogProbit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected


def test_DichotomousHill_13(ddataset):
    model = bmds2.models.DichotomousHill_13(ddataset)
    dfile = model.as_dfile()
    expected = "Dichotomous-Hill\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"
    assert dfile == expected
