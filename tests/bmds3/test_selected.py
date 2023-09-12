import bmds


class TestBmdsSelector:
    def test_selection(self, ddataset2):
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_model(bmds.constants.M_Logistic)
        session.execute_and_recommend()

        # show default undefined state
        assert session.selected.model is None
        assert session.selected.notes == ""
        assert session.selected.no_model_selected is True

        # accept recommendation when model recommended
        session.accept_recommendation()
        assert session.selected.model is session.models[0]
        assert session.selected.notes == "Selected as best-fitting model"
        assert session.selected.no_model_selected is False

        # accept recommendation when model recommended
        session.recommender.results.recommended_bmds_model_index = None
        session.accept_recommendation()
        assert session.selected.model is None
        assert session.selected.notes == "No model was selected as a best-fitting model"

        # show examples when a model is selected
        session.select(session.models[0], "best fitting")
        assert session.selected.model is session.models[0]
        assert session.selected.no_model_selected is False
        assert session.selected.serialize().dict() == dict(bmds_model_index=0, notes="best fitting")

        # show examples when a model is not selected
        session.select(None, "none")
        assert session.selected.model is None
        assert session.selected.no_model_selected is True
        assert session.selected.serialize().dict() == dict(bmds_model_index=None, notes="none")
