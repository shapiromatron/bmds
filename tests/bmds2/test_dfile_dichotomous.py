import bmds


def test_Logistic_213(ddataset):
    model = bmds.models.Logistic_213(ddataset)
    dfile = model.as_dfile()
    expected = "Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Logistic_214(ddataset):
    model = bmds.models.Logistic_214(ddataset)
    dfile = model.as_dfile()
    expected = "Logistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_LogLogistic_213(ddataset):
    model = bmds.models.LogLogistic_213(ddataset)
    dfile = model.as_dfile()
    expected = "LogLogistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_LogLogistic_214(ddataset):
    model = bmds.models.LogLogistic_214(ddataset)
    dfile = model.as_dfile()
    expected = "LogLogistic\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Gamma_215(ddataset):
    model = bmds.models.Gamma_215(ddataset)
    dfile = model.as_dfile()
    expected = "Gamma\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Gamma_216(ddataset):
    model = bmds.models.Gamma_216(ddataset)
    dfile = model.as_dfile()
    expected = "Gamma\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Probit_32(ddataset):
    model = bmds.models.Probit_32(ddataset)
    dfile = model.as_dfile()
    expected = "Probit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Probit_33(ddataset):
    model = bmds.models.Probit_33(ddataset)
    dfile = model.as_dfile()
    expected = "Probit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 0 0 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Multistage_32(ddataset):
    model = bmds.models.Multistage_32(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Multistage_33(ddataset):
    model = bmds.models.Multistage_33(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_MultistageCancer_19(ddataset):
    model = bmds.models.MultistageCancer_19(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage-Cancer\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_MultistageCancer_110(ddataset):
    model = bmds.models.MultistageCancer_110(ddataset)
    dfile = model.as_dfile()
    expected = "Multistage-Cancer\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4 2\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Weibull_215(ddataset):
    model = bmds.models.Weibull_215(ddataset)
    dfile = model.as_dfile()
    expected = "Weibull\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_Weibull_216(ddataset):
    model = bmds.models.Weibull_216(ddataset)
    dfile = model.as_dfile()
    expected = "Weibull\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_LogProbit_32(ddataset):
    model = bmds.models.LogProbit_32(ddataset)
    dfile = model.as_dfile()
    expected = "LogProbit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n250 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_LogProbit_33(ddataset):
    model = bmds.models.LogProbit_33(ddataset)
    dfile = model.as_dfile()
    expected = "LogProbit\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999\n0\n-9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected


def test_DichotomousHill_13(ddataset):
    model = bmds.models.DichotomousHill_13(ddataset)
    dfile = model.as_dfile()
    expected = "Dichotomous-Hill\nBMDS_Model_Run\n/temp/bmd/datafile.dax\n/temp/bmd/output.out\n4\n500 1e-08 1e-08 0 1 1 0 0\n0.1 0 0.95\n-9999 -9999 -9999 -9999\n0\n-9999 -9999 -9999 -9999\nDose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35"  # noqa
    assert dfile == expected
