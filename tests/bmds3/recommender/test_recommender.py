import pytest
from pydantic import ValidationError

import bmds
from bmds.bmds3.recommender.recommender import Recommender, RecommenderSettings

from ..run3 import RunBmds3


class TestRecommenderSettings:
    def test_default(self):
        # assert the default method actually works
        settings = RecommenderSettings.build_default()
        assert isinstance(settings, RecommenderSettings)

    def test_rule_validation(self):
        # assert that the entire rule list must be present
        settings = RecommenderSettings.build_default()
        settings.rules.pop()
        settings2 = settings.json()
        with pytest.raises(ValidationError) as err:
            RecommenderSettings.parse_raw(settings2)
        assert "Rule list must be complete" in str(err)


class TestRecommender:
    def test_df(self):
        df = Recommender().settings.to_df()
        assert df.shape == (22, 6)


@pytest.mark.skipif(not RunBmds3.should_run, reason=RunBmds3.skip_reason)
class TestSessionRecommender:
    def test_apply_logic_dich(self, ddataset2):
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_model(bmds.constants.M_DichotomousHill)
        session.add_model(bmds.constants.M_Gamma)
        session.execute_and_recommend()

        # get model bins
        assert session.recommender.results.model_bin == [0, 0]

        # model recommended and selection is accurate
        assert session.recommender.results.recommended_model_index == 1
        assert session.recommender.results.recommended_model_variable == "aic"
        assert session.models[1].results.fit.aic < session.models[0].results.fit.aic

    def test_apply_logic_cont(self, cdataset):
        session = bmds.session.Bmds330(dataset=cdataset)
        session.add_model(bmds.constants.M_Hill)
        session.add_model(bmds.constants.M_Power)
        session.execute_and_recommend()

        # get model bins
        assert session.recommender.results.model_bin == [1, 1]

        # model recommended and selection is accurate
        assert session.recommender.results.recommended_model_index is None
        assert session.recommender.results.recommended_model_variable is None
