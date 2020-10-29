from bmds.bmds3.constants import DichotomousModelChoices, Prior
from bmds.bmds3.types33 import DichotomousAnalysis


class TestDichotomousAnalysis:
    def test_priors_to_list(self, ddataset):
        # standard case; prior == num_params
        da = DichotomousAnalysis(
            model=DichotomousModelChoices.d_logistic.value,
            dataset=ddataset,
            priors=[
                Prior(type=0, initial_value=1, stdev=2, min_value=3, max_value=4),
                Prior(type=0, initial_value=1, stdev=2, min_value=3, max_value=4),
            ],
            BMD_type=1,
            BMR=0.1,
            alpha=0.05,
            degree=2,
            samples=100,
            burnin=20,
        )

        priors = da._priors_to_list()
        assert priors == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        assert len(priors) == len(da.priors) * 5

        # multistage case; len(priors) may be < num_params
        da.model = DichotomousModelChoices.d_multistage.value
        da.priors = [
            Prior(type=0, initial_value=1, stdev=1, min_value=1, max_value=1),
            Prior(type=0, initial_value=2, stdev=2, min_value=2, max_value=2),
            Prior(type=0, initial_value=3, stdev=3, min_value=3, max_value=3),
        ]

        # fmt: off
        da.degree = 2
        priors = da._priors_to_list()
        assert priors == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        assert len(priors) == (da.degree + 1) * 5

        da.degree = 3
        priors = da._priors_to_list()
        assert priors == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0]
        assert len(priors) == (da.degree + 1) * 5

        da.degree = 4
        priors = da._priors_to_list()
        assert priors == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 3.0]
        assert len(priors) == (da.degree + 1) * 5
        # fmt: on
