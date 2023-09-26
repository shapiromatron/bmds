from bmds.bmds3.models import nested_dichotomous


def test_execution(nd_dataset4):
    analysis = nested_dichotomous.NestedLogistic(nd_dataset4)
    analysis.execute()
    text = analysis.text()
    assert len(text) > 0
