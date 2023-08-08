from bmds.bmds3.models import nested_dichotomous
from bmds.session import Bmds330


def test_execution(nd_dataset):
    analysis = nested_dichotomous.NestedLogistic(nd_dataset)
    analysis.execute()


def test_session_execution(nd_dataset):
    session1 = Bmds330(dataset=nd_dataset)
    session1.add_default_models()
    session1.execute()
    # session1.execute_and_recommend() # TODO - implement recommend
    d = session1.to_dict()
    session2 = session1.from_serialized(d)
    assert session2.to_dict() == session1.to_dict()
