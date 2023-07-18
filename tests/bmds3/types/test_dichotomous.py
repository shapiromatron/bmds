from bmds.bmds3.models import dichotomous


class TestDichotomousAnalysisCPPStructs:
    def test_cpp_str(self, ddataset2):
        # ensure we can generate a string representation of the cpp structs
        model = dichotomous.Logistic(ddataset2)
        model.execute()
        text = str(model.structs)
        assert """- python_dichotomous_analysis""" in text
        assert """- python_dichotomous_model_result""" in text
        assert len(text.splitlines()) == 62
