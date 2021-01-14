import pytest

import bmds
from bmds.bmds2 import models


# dataset tests
@pytest.mark.mpl_image_compare
def test_cdataset_plot(cdataset):
    return cdataset.plot()


@pytest.mark.mpl_image_compare
def test_cidataset_plot(cidataset):
    return cidataset.plot()


@pytest.mark.mpl_image_compare
def test_ddataset_plot(ddataset):
    return ddataset.plot()


# test custom axes
@pytest.mark.mpl_image_compare
def test_cdataset_plot_customized():
    return bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
        xlabel="Dose (μg/m³)",
        ylabel="Relative liver weight (mg/kg)",
    ).plot()


@pytest.mark.mpl_image_compare
def test_cidataset_plot_customized():
    # fmt: off
    return bmds.ContinuousIndividualDataset(
        doses=[
            0, 0, 0, 0, 0, 0, 0, 0,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            1, 1, 1, 1, 1, 1,
            10, 10, 10, 10, 10, 10,
            100, 100, 100, 100, 100, 100,
            300, 300, 300, 300, 300, 300,
            500, 500, 500, 500, 500, 500,
        ],
        responses=[
            8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567,
            9.1556, 9.6821, 9.8256, 10.2095, 10.2222, 12.0382,
            9.5661, 9.7059, 9.9905, 10.2716, 10.471, 11.0602,
            8.8514, 10.0107, 10.0854, 10.5683, 11.1394, 11.4875,
            9.5427, 9.7211, 9.8267, 10.0231, 10.1833, 10.8685,
            10.368, 10.5176, 11.3168, 12.002, 12.1186, 12.6368,
            9.9572, 10.1347, 10.7743, 11.0571, 11.1564, 12.0368
        ],
        xlabel="Dose (μg/m³)",
        ylabel="Relative liver weight (mg/kg)",
    ).plot()
    # fmt: on


@pytest.mark.mpl_image_compare
def test_ddataset_plot_customized():
    return bmds.DichotomousDataset(
        doses=[0, 1.96, 5.69, 29.75],
        ns=[75, 49, 50, 49],
        incidences=[5, 1, 3, 14],
        xlabel="Dose (μg/m³)",
        ylabel="Fraction affected (%)",
    ).plot()


# continuous model tests
@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_polynomial_plot(cdataset):
    model = models.Polynomial_221(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_linear_plot(cdataset):
    model = models.Linear_221(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_exponential_m2_plot(cdataset):
    model = models.Exponential_M2_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_exponential_m3_plot(cdataset):
    model = models.Exponential_M3_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_exponential_m4_plot(cdataset):
    model = models.Exponential_M4_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_exponential_m5_plot(cdataset):
    model = models.Exponential_M5_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_power_plot(cdataset):
    model = models.Power_219(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_hill_plot(cdataset):
    model = models.Hill_218(cdataset)
    model.execute()
    return model.plot()


# dichotomous model tests
@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_multistage_plot(ddataset):
    model = models.Multistage_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_multistage_cancer_plot(ddataset):
    model = models.MultistageCancer_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_weibull_plot(ddataset):
    model = models.Weibull_217(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_logprobit_plot(ddataset):
    model = models.LogProbit_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_probit_plot(ddataset):
    model = models.Probit_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_gamma_plot(ddataset):
    model = models.Gamma_217(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_loglogistic_plot(ddataset):
    model = models.LogLogistic_215(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_logistic_plot(ddataset):
    model = models.Logistic_215(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
@pytest.mark.vcr()
def test_dichotomous_hill_plot(ddataset):
    model = models.DichotomousHill_13(ddataset)
    model.execute()
    return model.plot()
