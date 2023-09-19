import json
from pathlib import Path

import bmds
from bmds.bmds3 import BmdsSession


class TestBmds330:
    def test_serialization(self, ddataset2):
        # make sure serialize looks correct
        session1 = bmds.session.Bmds330(dataset=ddataset2)
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
        assert d["bmds_model_average"] is None
        # -> models recommendation
        assert d["recommender"]["settings"]["enabled"] is True
        assert d["recommender"]["results"]["recommended_model_variable"] == "aic"
        assert d["selected"]["bmds_model_index"] is None

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

    def test_serialization_ma(self, ddataset2, data_path, rewrite_data_files):
        # make sure serialize looks correct
        session1 = bmds.session.Bmds330(dataset=ddataset2)
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
        assert d["bmds_model_average"]["bmds_bmds_model_indexes"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert "bmd" in d["bmds_model_average"]["results"]

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

    def test_exports(self, ddataset2, rewrite_data_files):
        # make sure serialize looks correct
        session = bmds.session.Bmds330(dataset=ddataset2)
        session.add_default_models()
        session.execute_and_recommend()

        # dataframe
        df = session.to_df()

        # docx
        docx = session.to_docx(session_inputs_table=True)

        if rewrite_data_files:
            df.to_excel(Path("~/Desktop/bmds3-dichotomous.xlsx").expanduser(), index=False)
            docx.save(Path("~/Desktop/bmds3-dichotomous.docx").expanduser())

    def test_dll_version(self, ddataset2):
        session = bmds.session.Bmds330(dataset=ddataset2)
        version = session.dll_version()
        assert isinstance(version, str)
        assert int(version.split(".")[0]) >= 2021  # assume dll in format "YYYY.MM..."

    def test_citation(self, ddataset2):
        session = bmds.session.Bmds330(dataset=ddataset2)
        citations = session.citation()
        assert citations["paper"].startswith("Pham LL")
        assert citations["software"].startswith("Python BMDS.")
