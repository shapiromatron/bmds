from bmds.bmds3.models import nested_dichotomous


def test_execution(nd_dataset):
    analysis = nested_dichotomous.Logistic(nd_dataset)
    analysis.execute()

    print(analysis.results.dict())
    assert 1 == 0
