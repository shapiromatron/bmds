import unittest

import bmds

from .fixtures import *  # noqa


class TestBMDS(unittest.TestCase):

    def test_get_bmds_versions(self):
        versions = sorted(bmds.BMDS.get_versions())
        expected = sorted(['BMDS231', 'BMDS240', 'BMDS260', 'BMDS2601', 'BMDS270'])
        self.assertListEqual(versions, expected)

    def test_latest_bmds(self):
        session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS)
        assert session.version == bmds.constants.BMDS270

    def test_get_model(self):
        model = bmds.BMDS.get_model(bmds.constants.BMDS2601, bmds.constants.M_Probit)
        assert model == bmds.models.Probit_33


def test_default_model_additions(cdataset, ddataset):

    def num_polys(session):
        return len([m for m in session.models if m.model_name == 'Polynomial'])

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()
    assert len(session.models) == 10
    assert num_polys(session) == min(cdataset.num_dose_groups, 8) - 2

    for i in range(3, 9):
        array = range(i)
        print(array)
        ds = bmds.ContinuousDataset(doses=array, ns=array, means=array, stdevs=array)
        session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=ds)
        session.add_default_models()
        assert num_polys(session) == min(ds.num_dose_groups, 8) - 2


def test_group_models(cdataset):
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()

    # assert models are not grouped if no outputs exist
    assert len(session._group_models()) == len(session.models)

    # expected index output order ([1, 0], [2])
    session.models[0].output = {
        'BMD': 1,
        'AIC': 2,
        'parameters': [1, 2, 3]
    }
    session.models[1].output = {
        'BMD': 1,
        'AIC': 2,
        'parameters': [1, 2]
    }
    session.models[2].output = {
        'BMD': 3,
        'AIC': 2,
        'parameters': [1, 2, 3]
    }
    session.models = session.models[:3]
    groups = session._group_models()
    assert len(groups) == 2
    assert groups[0][0] == session.models[1]
    assert groups[0][1] == session.models[0]
    assert groups[1][0] == session.models[2]
