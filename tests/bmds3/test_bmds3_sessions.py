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
class TestBmds330:
    def test_serialization(self, dichds):
        # make sure serialize looks correct
        session1 = bmds.session.Bmds330(dataset=dichds)
        session1.add_default_models()
        session1.execute()
        serialized = session1.serialize()

        # spot check a few keys
        d = serialized.dict()
        assert d["version"]["numeric"] == bmds.session.Bmds330.version_tuple
        assert d["dataset"]["doses"] == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(d["models"]) == 10
        assert list(d["models"][0].keys()) == ["model_class", "settings", "results"]
        assert d["model_average"] is None

        # ensure we can convert back to a session
        session2 = serialized.deserialize()
        assert isinstance(session2, bmds.session.Bmds330)
        assert session2.dataset.doses == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(session2.models) == 10
        assert session2.models[0].has_output is True

        # make sure we get the same result back after deserializing
        assert session1.serialize().dict() == session2.serialize().dict()

    def test_serialization_ma(self, dichds):
        # make sure serialize looks correct
        session1 = bmds.session.Bmds330(dataset=dichds)
        session1.add_default_models()
        session1.add_model_averaging()
        session1.execute()
        serialized = session1.serialize()

        # spot check a few keys
        d = serialized.dict()
        assert d["version"]["numeric"] == bmds.session.Bmds330.version_tuple
        assert d["dataset"]["doses"] == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(d["models"]) == 10
        assert list(d["models"][0].keys()) == ["model_class", "settings", "results"]
        assert d["model_average"]["model_indexes"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert "bmd" in d["model_average"]["results"]

        # ensure we can convert back to a session
        session2 = serialized.deserialize()
        assert isinstance(session2, bmds.session.Bmds330)
        assert session2.dataset.doses == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(session2.models) == 10
        assert session2.models[0].has_output is True

        # make sure we get the same result back after deserializing
        assert session1.serialize().dict() == session2.serialize().dict()
