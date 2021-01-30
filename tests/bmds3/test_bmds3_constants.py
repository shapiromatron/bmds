import numpy as np
import pytest

from bmds.bmds3 import constants


@pytest.fixture
def mock_prior():
    return constants.ModelPriors(
        prior_class=constants.PriorClass.frequentist_restricted,
        priors=[
            constants.Prior(
                name="a",
                type=constants.PriorType.eNone,
                initial_value=1,
                stdev=1,
                min_value=1,
                max_value=1,
            ),
            constants.Prior(
                name="b",
                type=constants.PriorType.eNone,
                initial_value=2,
                stdev=2,
                min_value=2,
                max_value=2,
            ),
            constants.Prior(
                name="c",
                type=constants.PriorType.eNone,
                initial_value=3,
                stdev=3,
                min_value=3,
                max_value=3,
            ),
        ],
        variance_priors=[
            constants.Prior(
                name="d",
                type=constants.PriorType.eNone,
                initial_value=4,
                stdev=4,
                min_value=4,
                max_value=4,
            ),
            constants.Prior(
                name="e",
                type=constants.PriorType.eNone,
                initial_value=5,
                stdev=5,
                min_value=5,
                max_value=5,
            ),
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
            dist_type=constants.DistType.normal),
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        )
        assert np.allclose(
            mock_prior.to_c(dist_type=constants.DistType.log_normal),
            [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        )
        assert np.allclose(
            mock_prior.to_c(dist_type=constants.DistType.normal_ncv),
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )
        # fmt: on
