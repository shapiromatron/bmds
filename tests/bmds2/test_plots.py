import pytest

from bmds.bmds2 import models


# continuous model tests
@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_polynomial_plot(cdataset):
    model = models.Polynomial_221(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_linear_plot(cdataset):
    model = models.Linear_221(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_exponential_m2_plot(cdataset):
    model = models.Exponential_M2_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_exponential_m3_plot(cdataset):
    model = models.Exponential_M3_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_exponential_m4_plot(cdataset):
    model = models.Exponential_M4_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_exponential_m5_plot(cdataset):
    model = models.Exponential_M5_111(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_power_plot(cdataset):
    model = models.Power_219(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_hill_plot(cdataset):
    model = models.Hill_218(cdataset)
    model.execute()
    return model.plot()


# dichotomous model tests
@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_multistage_plot(ddataset):
    model = models.Multistage_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_multistage_cancer_plot(ddataset):
    model = models.MultistageCancer_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_weibull_plot(ddataset):
    model = models.Weibull_217(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_logprobit_plot(ddataset):
    model = models.LogProbit_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_probit_plot(ddataset):
    model = models.Probit_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_gamma_plot(ddataset):
    model = models.Gamma_217(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_loglogistic_plot(ddataset):
    model = models.LogLogistic_215(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_logistic_plot(ddataset):
    model = models.Logistic_215(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare(remove_text=True)
@pytest.mark.vcr()
def test_dichotomous_hill_plot(ddataset):
    model = models.DichotomousHill_13(ddataset)
    model.execute()
    return model.plot()
