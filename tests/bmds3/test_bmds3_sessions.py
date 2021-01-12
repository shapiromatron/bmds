import json
import os

import pytest

import bmds
from bmds.bmds3.sessions import BmdsSession

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
        session1.execute_and_recommend()
        d = session1.to_dict()

        # spot check a few keys
        # -> session metadata
        assert d["version"]["numeric"] == bmds.session.Bmds330.version_tuple
        # -> dataset
        assert d["dataset"]["doses"] == [0.0, 50.0, 100.0, 150.0, 200.0]
        # -> models (with results)
        assert len(d["models"]) == 10
        assert list(d["models"][0].keys()) == ["name", "model_class", "settings", "results"]
        # -> models average
        assert d["model_average"] is None
        # -> models recommendation
        assert d["recommender"]["settings"]["enabled"] is True
        assert d["recommender"]["results"]["recommended_model_variable"] == "aic"
        assert d["selected"]["model_index"] is None

        # ensure we can convert back to a session from JSON serialization
        session2 = BmdsSession.from_serialized(json.loads(json.dumps(d)))
        assert isinstance(session2, bmds.session.Bmds330)
        assert session2.dataset.doses == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(session2.models) == 10
        assert session2.models[0].has_results is True

        # make sure we get the same result back after deserializing
        d1 = session1.serialize().dict()
        d2 = session2.serialize().dict()
        assert d1 == d2

    def test_serialization_ma(self, dichds, data_path, rewrite_data_files):
        # make sure serialize looks correct
        session1 = bmds.session.Bmds330(dataset=dichds)
        session1.add_default_models()
        session1.add_model_averaging()
        session1.execute_and_recommend()
        d = session1.to_dict()

        if rewrite_data_files:
            (data_path / "dichotomous-session.json").write_text(d)

        # spot check a few keys
        assert d["version"]["numeric"] == bmds.session.Bmds330.version_tuple
        assert d["dataset"]["doses"] == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(d["models"]) == 10
        assert list(d["models"][0].keys()) == ["name", "model_class", "settings", "results"]
        assert d["model_average"]["model_indexes"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert "bmd" in d["model_average"]["results"]

        # ensure we can convert back to a session
        session2 = BmdsSession.from_serialized(json.loads(json.dumps(d)))
        assert isinstance(session2, bmds.session.Bmds330)
        assert session2.dataset.doses == [0.0, 50.0, 100.0, 150.0, 200.0]
        assert len(session2.models) == 10
        assert session2.models[0].has_results is True

        # make sure we get the same result back after deserializing
        d1 = session1.serialize().dict()
        d2 = session2.serialize().dict()
        assert d1 == d2
