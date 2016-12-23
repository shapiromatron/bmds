from .fixtures import *  # noqa

from bmds import models


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


# continuous model tests
@pytest.mark.mpl_image_compare
def test_polynomial_plot(cdataset):
    model = models.Polynomial_220(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_linear_plot(cdataset):
    model = models.Linear_220(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_exponential_m2_plot(cdataset):
    model = models.Exponential_M2_110(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_exponential_m3_plot(cdataset):
    model = models.Exponential_M3_110(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_exponential_m4_plot(cdataset):
    model = models.Exponential_M4_110(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_exponential_m5_plot(cdataset):
    model = models.Exponential_M5_110(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_power_plot(cdataset):
    model = models.Power_218(cdataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_hill_plot(cdataset):
    model = models.Hill_217(cdataset)
    model.execute()
    return model.plot()


# dichotomous model tests
@pytest.mark.mpl_image_compare
def test_multistage_plot(ddataset):
    model = models.Multistage_34(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_multistage_cancer_plot(ddataset):
    model = models.MultistageCancer_110(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_weibull_plot(ddataset):
    model = models.Weibull_216(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_logprobit_plot(ddataset):
    model = models.LogProbit_33(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_probit_plot(ddataset):
    model = models.Probit_33(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_gamma_plot(ddataset):
    model = models.Gamma_216(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_loglogistic_plot(ddataset):
    model = models.LogLogistic_214(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_logistic_plot(ddataset):
    model = models.Logistic_214(ddataset)
    model.execute()
    return model.plot()


@pytest.mark.mpl_image_compare
def test_dichotomous_hill_plot(ddataset):
    model = models.DichotomousHill_13(ddataset)
    model.execute()
    return model.plot()
