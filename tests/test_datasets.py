import pytest

import bmds
import numpy as np

from .fixtures import *  # noqa


def test_dataset_validation():
    # make dummy datasets
    dummy2 = [1, 2]
    dummy3 = [1, 2, 3]
    dummy4 = [1, 2, 3, 4]
    dummy3_dups = [0, 0, 1]

    # these should be valid
    bmds.DichotomousDataset(
        doses=dummy3, ns=dummy3, incidences=dummy3)
    bmds.DichotomousCancerDataset(
        doses=dummy2, ns=dummy2, incidences=dummy2)
    bmds.ContinuousDataset(
        doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3)
    bmds.ContinuousIndividualDataset(
        doses=dummy3, responses=dummy3)

    # these should raise errors
    with pytest.raises((IndexError, ValueError)):

        # insufficient number of dose groups
        bmds.DichotomousDataset(
            doses=dummy2, ns=dummy2, incidences=dummy2)
        bmds.ContinuousDataset(
            doses=dummy2, ns=dummy2, means=dummy2, stdevs=dummy2)

        # different sized lists
        bmds.DichotomousDataset(
            doses=dummy4, ns=dummy3, incidences=dummy3)
        bmds.ContinuousDataset(
            doses=dummy4, ns=dummy3, means=dummy3, stdevs=dummy3)
        bmds.ContinuousIndividualDataset(
            doses=dummy4, responses=dummy3)

        # also duplicate, but less than 2 dose-groups
        bmds.ContinuousIndividualDataset(
            doses=dummy3_dups, responses=dummy3)


def test_ci_summary_stats(cidataset):
    assert len(cidataset.doses) == 7
    assert np.isclose(
        cidataset.ns,
        [8, 6, 6, 6, 6, 6, 6]
    ).all()
    assert np.isclose(
        cidataset.means,
        [9.9264, 10.1889, 10.17755, 10.3571, 10.0275, 11.4933, 10.85275]
    ).all()
    assert np.isclose(
        cidataset.stdevs,
        [0.87969, 0.90166, 0.50089, 0.85590, 0.42833, 0.83734, 0.690373]
    ).all()


def test_dfile_outputs():
    dummy3 = [1, 2, 3]

    # check dichotomous
    ds = bmds.DichotomousDataset(
        doses=dummy3, ns=[5, 5, 5], incidences=[0, 1, 2])
    dfile = ds.as_dfile()
    expected = 'Dose Incidence NEGATIVE_RESPONSE\n1.000000 0 5\n2.000000 1 4\n3.000000 2 3'  # noqa
    assert dfile == expected

    # check continuous
    ds = bmds.ContinuousDataset(
        doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3)
    dfile = ds.as_dfile()
    expected = 'Dose NumAnimals Response Stdev\n1.000000 1 1.000000 1.000000\n2.000000 2 2.000000 2.000000\n3.000000 3 3.000000 3.000000'  # noqa
    assert dfile == expected

    # check continuous individual
    ds = bmds.ContinuousIndividualDataset(
        doses=dummy3, responses=dummy3)
    dfile = ds.as_dfile()
    expected = 'Dose Response\n1.000000 1.000000\n2.000000 2.000000\n3.000000 3.000000'  # noqa
    assert dfile == expected


def test_is_increasing():
    dummy4 = [1, 2, 3, 4]

    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4,
                                means=dummy4, stdevs=dummy4)
    assert ds.is_increasing is True

    rev = list(reversed(dummy4))
    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4,
                                means=rev, stdevs=dummy4)
    assert ds.is_increasing is False

    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4,
                                means=[1, 2, 3, 0], stdevs=dummy4)
    assert ds.is_increasing is True

    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4,
                                means=[1, 3, 2, 1], stdevs=dummy4)
    assert ds.is_increasing is True

    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4,
                                means=[0, 2, -1, 0], stdevs=dummy4)
    assert ds.is_increasing is True


def test_dose_drops(cidataset):

    cdataset = bmds.ContinuousDataset(
        doses=list(reversed([0, 10, 50, 150, 400])),
        ns=list(reversed([111, 142, 143, 93, 42])),
        means=list(reversed([2.112, 2.095, 1.956, 1.587, 1.254])),
        stdevs=list(reversed([0.235, 0.209, 0.231, 0.263, 0.159])))

    assert cdataset.as_dfile() == 'Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000'  # noqa
    cdataset.drop_dose()
    assert cdataset.as_dfile() == 'Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000'  # noqa
    cdataset.drop_dose()
    assert cdataset.as_dfile() == 'Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000'  # noqa
    with pytest.raises(ValueError):
        cdataset.drop_dose()

    ddataset = bmds.DichotomousDataset(
        doses=list(reversed([0, 1.96, 5.69, 29.75])),
        ns=list(reversed([75, 49, 50, 49])),
        incidences=list(reversed([5, 1, 3, 14])))
    assert ddataset.as_dfile() == 'Dose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47\n29.750000 14 35'  # noqa
    ddataset.drop_dose()
    assert ddataset.as_dfile() == 'Dose Incidence NEGATIVE_RESPONSE\n0.000000 5 70\n1.960000 1 48\n5.690000 3 47'  # noqa
    with pytest.raises(ValueError):
        ddataset.drop_dose()

    assert cidataset.as_dfile() == 'Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500\n300.000000 10.368000\n300.000000 10.517600\n300.000000 11.316800\n300.000000 12.002000\n300.000000 12.118600\n300.000000 12.636800\n500.000000 9.957200\n500.000000 10.134700\n500.000000 10.774300\n500.000000 11.057100\n500.000000 11.156400\n500.000000 12.036800'  # noqa
    cidataset.drop_dose()
    assert cidataset.as_dfile() == 'Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500\n300.000000 10.368000\n300.000000 10.517600\n300.000000 11.316800\n300.000000 12.002000\n300.000000 12.118600\n300.000000 12.636800'  # noqa
    cidataset.drop_dose()
    assert cidataset.as_dfile() == 'Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500'  # noqa
    cidataset.drop_dose()
    cidataset.drop_dose()
    with pytest.raises(ValueError):
        cidataset.drop_dose()


def test_anova(anova_dataset, bad_anova_dataset):
    # Check that anova generates expected output from original specifications.
    report = anova_dataset.get_anova_report()
    expected = '                     Tests of Interest    \n   Test    -2*log(Likelihood Ratio)  Test df        p-value    \n   Test 1              22.2699         12           0.0346\n   Test 2               5.5741          6           0.4725\n   Test 3               5.5741          6           0.4725'  # noqa
    assert report == expected

    # check bad anova dataset
    report = bad_anova_dataset.get_anova_report()
    expected = 'ANOVA cannot be calculated for this dataset.'
    assert report == expected


def test_correct_variance_model(cdataset):
    # Check that the correct variance model is selected for dataset
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    for model in session.model_options:
        session.add_model(bmds.constants.M_Power)
    session.execute()
    model = session.models[0]
    calc_pvalue2 = cdataset.anova[1].TEST
    correct_pvalue2 = model.output['p_value2']
    # large tolerance due to reporting in text-file
    atol = 1e-3
    assert np.isclose(calc_pvalue2, correct_pvalue2, atol=atol)


def test_extra_kwargs(cdataset, cidataset, ddataset):
    ds = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
        id='123')

    assert 'id' in ds.to_dict()

    ds = bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14],
        id=123)

    assert 'id' in ds.to_dict()

    ds = bmds.ContinuousIndividualDataset(
        doses=[
            0, 0, 1, 1, 2, 2, 3, 3,
        ],
        responses=[
            8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567,
        ],
        id=None)

    assert 'id' in ds.to_dict()


def test_dataset_reporting_options(cdataset):
    # test defaults
    assert cdataset._get_dose_units_text() == ''
    assert cdataset._get_response_units_text() == ''
    assert cdataset._get_dataset_name() == 'BMDS output results'

    # test overrides
    ds = bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
        dose_units='μg/m³',
        response_units='mg/kg',
        dataset_name='Smith 2017: Relative Liver Weight in Male SD Rats',
    )
    assert ds._get_dataset_name() == 'Smith 2017: Relative Liver Weight in Male SD Rats'
    assert ds._get_dose_units_text() == ' (μg/m³)'
    assert ds._get_response_units_text() == ' (mg/kg)'
