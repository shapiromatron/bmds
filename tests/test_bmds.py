import unittest

import bmds


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
