import unittest

import bmds

from .fixtures import *


class TestBMDS(unittest.TestCase):

    def test_get_bmds_versions(self):
        versions = sorted(bmds.BMDS.get_versions())
        expected = sorted(['BMDS231', 'BMDS240', 'BMDS260', 'BMDS2601'])
        self.assertListEqual(versions, expected)

    def test_latest_bmds(self):
        session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS)
        assert session.version == bmds.constants.BMDS2601

    def test_get_model(self):
        model = bmds.BMDS.get_model(bmds.constants.BMDS2601, bmds.constants.M_Probit)
        assert model == bmds.models.Probit_33


def test_default_model_additions(cdataset, ddataset):

    def num_polys(session):
        return len([m for m in session.models if m.model_name == 'Polynomial'])

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()
    assert len(session.models) == 10
    assert num_polys(session) == min(cdataset.num_doses, 8) - 2

    for i in range(3, 9):
        array = range(i)
        print(array)
        ds = bmds.ContinuousDataset(doses=array, ns=array, means=array, stdevs=array)
        session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=ds)
        session.add_default_models()
        assert num_polys(session) == min(ds.num_doses, 8) - 2
