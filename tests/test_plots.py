from io import BytesIO
import os

from .fixtures import *  # noqa


RESOURCES = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))


def test_dataset_plots(cdataset, cidataset, ddataset):
    tests = (
        (cdataset, 'cdataset.png'),
        (cidataset, 'cidataset.png'),
        (ddataset, 'ddataset.png'),
    )
    for ds, fn in tests:
        plot = ds.plot()
        f = BytesIO()
        plot.savefig(f, format='png')

        actual = f.getvalue()
        f.close()

        with open(os.path.join(RESOURCES, fn), 'rb') as f:
            expected = f.read()

        assert actual == expected
