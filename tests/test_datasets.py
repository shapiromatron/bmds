import pytest

import bmds


def test_dataset_validation():
    # make dummy datasets
    dummy3 = [1, 2, 3]
    dummy4 = [1, 2, 3, 4]

    # these should be valid
    bmds.DichotomousDataset(
        doses=dummy3, ns=dummy3, incidences=dummy3)
    bmds.ContinuousDataset(
        doses=dummy3, ns=dummy3, responses=dummy3, stdevs=dummy3)

    # these should raise errors
    with pytest.raises(ValueError):
        # different sized lists
        bmds.DichotomousDataset(
            doses=dummy4, ns=dummy3, incidences=dummy3)
        bmds.ContinuousDataset(
            doses=dummy4, ns=dummy3, responses=dummy3, stdevs=dummy3)

        # 2 remaining after dropping-doses
        bmds.DichotomousDataset(
            doses=dummy3, ns=dummy3, incidences=dummy3,
            doses_dropped=1)
        bmds.ContinuousDataset(
            doses=dummy3, ns=dummy3, responses=dummy3, stdevs=dummy3,
            doses_dropped=1)


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
        doses=dummy4, ns=dummy4, responses=dummy4, stdevs=dummy4,
        doses_dropped=1)
    dfile = ds.as_dfile()
    expected = 'Dose NumAnimals Response Stdev\n1.000000 1.000000 1.000000 1.000000\n2.000000 2.000000 2.000000 2.000000\n3.000000 3.000000 3.000000 3.000000'  # noqa
    assert dfile == expected


def test_doses_used():
    ds5 = [1, 2, 3, 4, 5]

    ds = bmds.DichotomousDataset(ds5, ds5, ds5)
    assert ds.doses_used == 5
    ds = bmds.DichotomousDataset(ds5, ds5, ds5, doses_dropped=2)
    assert ds.doses_used == 3

    ds = bmds.ContinuousDataset(ds5, ds5, ds5, ds5)
    assert ds.doses_used == 5
    ds = bmds.ContinuousDataset(ds5, ds5, ds5, ds5, doses_dropped=2)
    assert ds.doses_used == 3
