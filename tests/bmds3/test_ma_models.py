import json

import numpy as np
import pytest

import bmds
from bmds.bmds3.constants import PriorClass

from .run3 import RunBmds3


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
class TestDichotomousMa:
    def test_bmds3_dichotomous_ma_session(self, ddataset2):
        # check execution and it can be json serialized
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_default_bayesian_models()
        session.execute()
        d = session.to_dict()
        assert isinstance(json.dumps(d), str)

    def test_prior_weights(self, ddataset2):
        # default; equal weights
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_model(bmds.constants.M_Logistic, {"priors": PriorClass.bayesian})
        session.add_model(bmds.constants.M_Probit, {"priors": PriorClass.bayesian})
        session.add_model_averaging()
        assert np.allclose(session.ma_weights, [0.5, 0.5])
        session.execute()
        assert np.allclose(session.model_average.results.priors, [0.5, 0.5])
        assert np.allclose(session.model_average.results.posteriors, [0.065, 0.94], atol=0.05)

        # custom; propagate through results
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_model(bmds.constants.M_Logistic, {"priors": PriorClass.bayesian})
        session.add_model(bmds.constants.M_Probit, {"priors": PriorClass.bayesian})
        session.add_model_averaging(weights=[0.9, 0.1])
        assert np.allclose(session.ma_weights, [0.9, 0.1])
        session.execute()
        assert np.allclose(session.model_average.results.priors, [0.9, 0.1])
        assert np.allclose(session.model_average.results.posteriors, [0.38, 0.62], atol=0.05)
