from contextlib import contextmanager
from copy import Error

from . import db, schemas


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    try:
        yield db.session
        db.session.commit()
    except Error:
        db.session.rollback()
    finally:
        db.session.close()


def _execute_bmds270_model(model, model_name, dataset, version):
    model.execute()
    if model.has_successfully_executed:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id,
            bmds_version=version,
            model=model_name,
            completed=True,
            inputs=model.get_default(),
            outputs=model.output,
            bmd=model.output["BMD"],
            bmdl=model.output["BMDL"],
            bmdu=model.output["BMDU"],
            aic=model.output["AIC"],
        )
    else:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id, bmds_version=version, model=model_name, completed=False
        )


def _execute_bmds330_model(model, model_name, dataset, version):
    model.execute()
    if model.has_results:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id,
            bmds_version=version,
            model=model_name,
            completed=True,
            inputs=model.settings.dict(),
            outputs=model.results.dict(),
            bmd=model.results.bmd,
            bmdl=model.results.bmdl,
            bmdu=model.results.bmdu,
            aic=model.results.fit.aic,
        )
    else:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id, bmds_version=version, model=model_name, completed=False
        )
