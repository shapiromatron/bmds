import unittest

import bmds


class TestVersionsAndModels(unittest.TestCase):

    def test_get_versions(self):
        versions = bmds.get_versions()
        expected = ['2.40', '2.31', '2.30']
        self.assertListEqual(versions, expected)

    def test_get_models(self):
        models = bmds.get_models_for_version('2.40')
        dich_model_names = models['D'].keys()
        expected = [
            'LogLogistic', 'Weibull', 'Probit',
            'LogProbit', 'Logistic', 'Gamma', 'Multistage']
        self.assertListEqual(dich_model_names, expected)
