import bmds

from .fixtures import *  # noqa


def test_batch(cdataset, ddataset):

    batch = bmds.SessionBatch()

    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    batch.append(session)

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS, dataset=ddataset)
    session.add_default_models()
    session.execute()
    batch.append(session)

    session = bmds.BMDS.latest_version(bmds.constants.DICHOTOMOUS_CANCER, dataset=ddataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    batch.append(session)

    # check input options
    df1 = batch.to_df(include_io=True, recommended_only=True)
    df2 = batch.to_df(include_io=False, recommended_only=False)

    # check include_io
    assert 'dfile' in df1.columns and 'dfile' not in df2.columns
    assert 'outfile' in df1.columns and 'outfile' not in df2.columns

    # check recommended_only
    assert df1.shape[0] == 3 and df2.shape[0] == 23
