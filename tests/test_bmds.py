import unittest

import bmds


class TestVersionsAndModels(unittest.TestCase):

    def test_get_bmds_versions(self):
        versions = sorted(bmds.get_bmds_versions())
        expected = sorted(['BMDS231', 'BMDS240', 'BMDS260', 'BMDS2601'])
        self.assertListEqual(versions, expected)

    def test_latest_bmds(self):
        session = bmds.latest_bmds(bmds.constants.CONTINUOUS)
        assert session.version == bmds.constants.BMDS2601
