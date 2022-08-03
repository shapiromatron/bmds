import numpy as np
import pytest

from bmds.bmds3.constants import DistType, PriorClass, PriorType
from bmds.bmds3.types.priors import ModelPriors, Prior


@pytest.fixture
def mock_prior():
    t = PriorType.Uniform
    return ModelPriors(
        prior_class=PriorClass.frequentist_restricted,
        priors=[
            Prior(name="a", type=t, initial_value=1, stdev=1, min_value=1, max_value=1),
            Prior(name="b", type=t, initial_value=2, stdev=2, min_value=2, max_value=2),
            Prior(name="c", type=t, initial_value=3, stdev=3, min_value=3, max_value=3),
        ],
        variance_priors=[
            Prior(name="d", type=t, initial_value=4, stdev=4, min_value=4, max_value=4),
            Prior(name="e", type=t, initial_value=5, stdev=5, min_value=5, max_value=5),
        ],
    )


class TestModelPriors:
    def test_to_c(self, mock_prior):
        # fmt: off
        assert np.allclose(
            mock_prior.to_c(),
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        )
        assert np.allclose(
            mock_prior.to_c(degree=1),
            [0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        )
        assert np.allclose(
            mock_prior.to_c(degree=2),
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        )
        assert np.allclose(
            mock_prior.to_c(degree=3),
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0]
        )
        assert np.allclose(
            mock_prior.to_c(degree=4),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0]
        )
        assert np.allclose(mock_prior.to_c(
            dist_type=DistType.normal),
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0]
        )
        assert np.allclose(
            mock_prior.to_c(dist_type=DistType.log_normal),
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0, 1.0, 2.0, 3.0, 5.0]
        )
        assert np.allclose(
            mock_prior.to_c(dist_type=DistType.normal_ncv),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )
        # fmt: on
