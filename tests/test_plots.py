from .fixtures import *  # noqa


@pytest.mark.mpl_image_compare
def test_cdataset_plot(cdataset):
    return cdataset.plot()


@pytest.mark.mpl_image_compare
def test_cidataset_plot(cidataset):
    return cidataset.plot()


@pytest.mark.mpl_image_compare
def test_ddataset_plot(ddataset):
    return ddataset.plot()

