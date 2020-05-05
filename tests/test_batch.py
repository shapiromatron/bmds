import numpy as np
import pytest

import bmds

from .fixtures import *  # noqa


@pytest.mark.vcr()
def test_batch(cdataset, ddataset):

    batch = bmds.SessionBatch()

    session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    batch.append(session)

    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS, dataset=ddataset)
    session.add_default_models()
    session.execute()
    batch.append(session)

    session = bmds.BMDS.version("BMDS270", bmds.constants.DICHOTOMOUS_CANCER, dataset=ddataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    batch.append(session)

    # check input options
    df1 = batch.to_df(include_io=True, recommended_only=True)
    df2 = batch.to_df(include_io=False, recommended_only=False)

    # check include_io
    assert "dfile" in df1.columns and "dfile" not in df2.columns
    assert "outfile" in df1.columns and "outfile" not in df2.columns

    # check recommended_only
    assert df1.shape[0] == 3 and df2.shape[0] == 23

    # check that a list of dictionaries is successfully built
    dictionaries = batch.to_dicts()
    assert len(dictionaries) == 3

    # check dataset export, model exported, and recommendation is correct.
    first = dictionaries[0]
    assert first["dataset"]["doses"][-1] == 400
    assert first["recommended_model_index"] is None
    assert np.isclose(first["models"][0]["output"]["BMD"], 99.9419)
