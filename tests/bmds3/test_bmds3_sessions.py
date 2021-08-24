import json
import os
from pathlib import Path

import pytest

import bmds
from bmds import constants
from bmds.bmds3 import BmdsSession, BmdsSessionBatch

# TODO remove this restriction
should_run = os.getenv("CI") is None
skip_reason = "DLLs not present on CI"


@pytest.fixture
def dichds():
    return bmds.DichotomousDataset(
        doses=[0, 50, 100, 150, 200], ns=[100, 100, 100, 100, 100], incidences=[0, 5, 30, 65, 90]
    )


@pytest.fixture
def contds():
    return bmds.ContinuousDataset(
        doses=[0, 50, 100, 150, 200],
        ns=[100, 100, 100, 100, 100],
        means=[10, 20, 30, 40, 50],
        stdevs=[3, 4, 5, 6, 7],
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
            (data_path / "dichotomous-session.json").write_text(session1.serialize().json())

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

    def test_exports(self, dichds, rewrite_data_files):
        # make sure serialize looks correct
        session = bmds.session.Bmds330(dataset=dichds)
        session.add_default_models()
        session.execute_and_recommend()

        # dataframe
        df = session.to_df()

        # docx
        docx = session.to_docx()

        if rewrite_data_files:
            df.to_excel(Path("~/Desktop/bmds3-dichotomous.xlsx").expanduser(), index=False)
            docx.save(Path("~/Desktop/bmds3-dichotomous.docx").expanduser())


@pytest.mark.skipif(not should_run, reason=skip_reason)
class TestBmdsSessionBatch:
    def test_exports(self, dichds, contds, rewrite_data_files):
        # datasets = [dichds, contds]

        datasets = [dichds]
        batch = BmdsSessionBatch()
        for dataset in datasets:
            session = bmds.session.Bmds330(dataset=dataset)
            session.add_default_models()
            session.execute_and_recommend()
            batch.sessions.append(session)

        # check serialization/deserialization
        data = batch.serialize()
        batch2 = batch.deserialize(data)
        assert len(batch2.sessions) == len(batch.sessions)

        # check exports
        df = batch.to_df()
        docx = batch.to_docx()

        if rewrite_data_files:
            df.to_excel(Path("~/Desktop/bmds3-batch.xlsx").expanduser(), index=False)
            docx.save(Path("~/Desktop/bmds3-batch.docx").expanduser())

    def test_exports_ci(self, cidataset, rewrite_data_files):
        # datasets = [dichds, contds]

        datasets = [cidataset]
        batch = BmdsSessionBatch()
        for dataset in datasets:
            session = bmds.session.Bmds330(dataset=dataset)
            # session.add_default_models()
            session.add_model(constants.M_Power)
            session.execute_and_recommend()
            batch.sessions.append(session)

        # check serialization/deserialization
        data = batch.serialize()
        batch2 = batch.deserialize(data)
        assert len(batch2.sessions) == len(batch.sessions)

        # check exports
        df = batch.to_df()
        docx = batch.to_docx()

        df.to_excel(Path("~/Desktop/bmds3-batch.xlsx").expanduser(), index=False)
        docx.save(Path("~/Desktop/bmds3-batch.docx").expanduser())

    def test_exports_cs(self, contds, rewrite_data_files):
        # datasets = [dichds, contds]

        datasets = [contds]
        batch = BmdsSessionBatch()
        for dataset in datasets:
            session = bmds.session.Bmds330(dataset=dataset)
            # session.add_default_models()
            session.add_model(constants.M_Power)
            session.execute_and_recommend()
            batch.sessions.append(session)

        # check serialization/deserialization
        data = batch.serialize()
        batch2 = batch.deserialize(data)
        assert len(batch2.sessions) == len(batch.sessions)

        # check exports
        df = batch.to_df()
        docx = batch.to_docx()

        df.to_excel(Path("~/Desktop/bmds3-batch.xlsx").expanduser(), index=False)
        docx.save(Path("~/Desktop/bmds3-batch.docx").expanduser())
