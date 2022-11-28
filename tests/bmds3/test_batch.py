import tempfile
from pathlib import Path

import bmds
from bmds import constants
from bmds.bmds3.batch import BmdsSessionBatch, ExecutionResponse
from bmds.bmds3.sessions import Bmds330


class TestBmdsSessionBatch:
    def test_execute(self, ddataset2):
        def runner(ds):
            sess = Bmds330(dataset=ds)
            sess.add_model(constants.M_Logistic)
            sess.execute()
            return ExecutionResponse(success=True, content=[sess.to_dict()])

        batch = BmdsSessionBatch.execute([ddataset2], runner, nprocs=1)
        assert len(batch.sessions) == 1

    def test_exports_dichotomous(self, ddataset2, rewrite_data_files):
        datasets = [ddataset2]
        batch = BmdsSessionBatch()
        for dataset in datasets:
            session = bmds.session.Bmds330(dataset=dataset)
            session.add_default_models()
            session.execute_and_recommend()
            batch.sessions.append(session)

            session = bmds.session.Bmds330(dataset=dataset)
            session.add_default_bayesian_models()
            session.execute()
            batch.sessions.append(session)

        # check serialization/deserialization
        data = batch.serialize()
        batch2 = batch.deserialize(data)
        assert len(batch2.sessions) == len(batch.sessions)

        # check zip
        zf = Path(tempfile.NamedTemporaryFile().name)
        try:
            # save
            batch.save(zf)
            assert zf.exists()
            # unsave
            batch3 = BmdsSessionBatch.load(zf)
            assert len(batch3.sessions) == 2
        finally:
            zf.unlink()

        # check exports
        excel = batch.to_excel()
        docx = batch.to_docx()

        if rewrite_data_files:
            Path("~/Desktop/bmds3-d-batch.xlsx").expanduser().write_bytes(excel.getvalue())
            docx.save(Path("~/Desktop/bmds3-d-batch.docx").expanduser())

    def test_exports_continuous(self, cdataset2, cidataset, rewrite_data_files):
        datasets = [cdataset2, cidataset]
        batch = BmdsSessionBatch()
        for dataset in datasets:
            session = bmds.session.Bmds330(dataset=dataset)
            session.add_model(constants.M_Power)
            session.execute_and_recommend()
            batch.sessions.append(session)

        # check serialization/deserialization
        data = batch.serialize()
        batch2 = batch.deserialize(data)
        assert len(batch2.sessions) == len(batch.sessions)

        # check exports
        excel = batch.to_excel()
        docx = batch.to_docx()

        if rewrite_data_files:
            Path("~/Desktop/bmds3-c-batch.xlsx").expanduser().write_bytes(excel.getvalue())
            docx.save(Path("~/Desktop/bmds3-c-batch.docx").expanduser())
