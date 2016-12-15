import pytest

from .fixtures import *
import bmds


def test_batch(cdataset, ddataset):

    batch = bmds.SessionBatch()

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_model(bmds.constants.M_Power)
    session.execute()
    session.recommend()
    batch.append(session)

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    session.add_model(bmds.constants.M_Logistic)
    session.execute()
    session.recommend()
    batch.append(session)

    with pytest.raises(NotImplementedError):
        batch.to_df()
        batch.to_csv('~/Desktop/output.csv')
        batch.to_png('~/Desktop')
