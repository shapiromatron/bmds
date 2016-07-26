import unittest

import bmds


class TestVersionsAndModels(unittest.TestCase):

    def test_get_versions(self):
        versions = sorted(bmds.get_versions())
        expected = sorted(['BMDS230', 'BMDS231', 'BMDS240', 'BMDS260', 'BMDS2601'])
        self.assertListEqual(versions, expected)

    def test_get_models(self):
        models = bmds.get_models_for_version('BMDS240')
        dich_model_names = models['D'].keys()
        expected = [
            'Logistic', 'LogLogistic', 'Probit',
            'LogProbit', 'Multistage', 'Gamma', 'Weibull',
        ]
        self.assertListEqual(dich_model_names, expected)
