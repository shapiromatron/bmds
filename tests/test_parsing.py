# -*- coding: utf-8 -*-
import bmds

from .fixtures import *  # noqa


expected_bad_anova_warnings = """THIS USUALLY MEANS THE MODEL HAS NOT CONVERGED!
BMDL computation failed.
Warning:  optimum may not have been found.  Bad completion code in Optimization routine.
Warning: Likelihood for fitted model larger than the Likelihood for model A3."""


def test_bad_anova_parsing(bad_anova_dataset):
    with open('./tests/outfiles/bad_anova_power.out', 'r') as f:
        outfile = f.read()
    model = bmds.models.Power_218(bad_anova_dataset)
    model.parse_results(outfile)
    warnings = model.output['warnings']
    actual = '\n'.join(warnings)
    assert expected_bad_anova_warnings == actual


def test_bad_anova_parsing2(bad_anova_dataset):
    with open('./tests/outfiles/bad_anova_hill.out', 'r') as f:
        outfile = f.read()
    model = bmds.models.Hill_217(bad_anova_dataset)
    model.parse_results(outfile)
    assert model.outfile['Chi2'] == '1.#QNAN'
