import pytest

import bmds
import numpy as np

from .fixtures import *


def test_dataset_validation():
    # make dummy datasets
    dummy3 = [1, 2, 3]
    dummy4 = [1, 2, 3, 4]
    dummy3_dups = [0, 0, 1]

    # these should be valid
    bmds.DichotomousDataset(
        doses=dummy3, ns=dummy3, incidences=dummy3)
    bmds.ContinuousDataset(
        doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3)
    bmds.ContinuousIndividualDataset(
        doses=dummy3, responses=dummy3)

    # these should raise errors
    with pytest.raises(ValueError):
        # different sized lists
        bmds.DichotomousDataset(
            doses=dummy4, ns=dummy3, incidences=dummy3)
        bmds.ContinuousDataset(
            doses=dummy4, ns=dummy3, means=dummy3, stdevs=dummy3)
        bmds.ContinuousIndividualDataset(
            doses=dummy4, responses=dummy3)

        # 2 remaining after dropping-doses
        bmds.DichotomousDataset(
            doses=dummy3, ns=dummy3, incidences=dummy3,
            doses_dropped=1)
        bmds.ContinuousDataset(
            doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3,
            doses_dropped=1)
        bmds.ContinuousIndividualDataset(
            doses=dummy3, responses=dummy3,
            doses_dropped=1)

        # duplicate dose-groups
        bmds.DichotomousDataset(
            doses=dummy3_dups, ns=dummy3, incidences=dummy3,
            doses_dropped=1)
        bmds.ContinuousDataset(
            doses=dummy3_dups, ns=dummy3, means=dummy3, stdevs=dummy3,
            doses_dropped=1)

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
    dummy4 = [1, 2, 3, 4]

    # check dichotomous
    ds = bmds.DichotomousDataset(
        doses=dummy4, ns=[5, 5, 5, 5], incidences=[0, 1, 2, 3],
        doses_dropped=1)
    dfile = ds.as_dfile()
    expected = 'Dose Incidence NEGATIVE_RESPONSE\n1.000000 0 5\n2.000000 1 4\n3.000000 2 3'  # noqa
    assert dfile == expected

    # check continuous
    ds = bmds.ContinuousDataset(
        doses=dummy4, ns=dummy4, means=dummy4, stdevs=dummy4,
        doses_dropped=1)
    dfile = ds.as_dfile()
    expected = 'Dose NumAnimals Response Stdev\n1.000000 1 1.000000 1.000000\n2.000000 2 2.000000 2.000000\n3.000000 3 3.000000 3.000000'  # noqa
    assert dfile == expected

    # check continuous individual
    ds = bmds.ContinuousIndividualDataset(
        doses=dummy4, responses=dummy4,
        doses_dropped=1)
    dfile = ds.as_dfile()
    expected = 'Dose Response\n1.000000 1.000000\n2.000000 2.000000\n3.000000 3.000000'  # noqa
    assert dfile == expected


def test_doses_used():
    ds5 = [1, 2, 3, 4, 5]
    ds5dups = [1, 2, 3, 4, 5] * 2

    ds = bmds.DichotomousDataset(ds5, ds5, ds5)
    assert ds.doses_used == 5
    ds = bmds.DichotomousDataset(ds5, ds5, ds5, doses_dropped=2)
    assert ds.doses_used == 3

    ds = bmds.ContinuousDataset(ds5, ds5, ds5, ds5)
    assert ds.doses_used == 5
    ds = bmds.ContinuousDataset(ds5, ds5, ds5, ds5, doses_dropped=2)
    assert ds.doses_used == 3

    ds = bmds.ContinuousIndividualDataset(ds5dups, ds5dups)
    assert ds.doses_used == 5
    ds = bmds.ContinuousIndividualDataset(ds5dups, ds5dups, doses_dropped=2)
    assert ds.doses_used == 3


def test_is_increasing():
    dummy4 = [1, 2, 3, 4]

    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=dummy4, stdevs=dummy4)
    assert ds.is_increasing is True

    rev = list(reversed(dummy4))
    ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=rev, stdevs=dummy4)
    assert ds.is_increasing is False


def test_anova(anova_dataset):
    # Check that anova generates expected output from original specifications.
    report = anova_dataset.get_anova_report()
    expected = '                     Tests of Interest    \n   Test    -2*log(Likelihood Ratio)  Test df        p-value    \n   Test 1              22.2699         12           0.0346\n   Test 2               5.5741          6           0.4725\n   Test 3               5.5741          6           0.4725'  # noqa
    assert report == expected


def test_correct_variance_model(cdataset):
    # TODO - check for constant and non constant datasets
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
