import os

import pytest

import bmds

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.mark.skipif(not should_run, reason=skip_reason)
class TestBmdsSelector:
    def test_selection(self, dichds):
        session = bmds.session.Bmds330(dataset=dichds)
        session.add_model(bmds.constants.M_Logistic)
        session.execute_and_recommend()

        # show default undefined state
        assert session.selected.model is None
        assert session.selected.no_model_selected is False

        # show examples when a model is selected
        session.selected.select(session.models[0], "best fitting")
        assert session.selected.model is session.models[0]
        assert session.selected.no_model_selected is False
        expected = dict(model_index=0, notes="best fitting")
        assert session.selected.serialize().dict() == expected

        # show examples when a model is not selected
        session.selected.select(None, "no model selected")
        assert session.selected.model is None
        assert session.selected.no_model_selected is True
        assert session.selected.serialize().dict() == {
            "model_index": None,
            "notes": "no model selected",
        }
