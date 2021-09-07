from contextlib import contextmanager

from . import db
from . import schemas




@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    try:
        yield db.session
        db.session.commit()
    except:
        db.session.rollback()
        raise
    finally:
        db.session.close()

def _execute_bmds2_model(model,model_name,dataset,version):
    model.execute()
    if model.has_successfully_executed:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id,
            bmds_version=version,
            model=model_name,
            completed=True,
            inputs=model.get_default(),
            outputs=model.output,
            bmd=model.output['BMD'],
            bmdl=model.output["BMDL"],
            bmdu=model.output["BMDU"],
            aic=model.output["AIC"]
        )
    else:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id,
            bmds_version=version,
            model=model_name,
            completed=False
        )

def _execute_bmds3_model(model,model_name,dataset,version):
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
            aic=model.results.fit.aic
        )
    else:
        return schemas.ResultSchema(
            dataset_id=dataset.metadata.id,
            bmds_version=version,
            model=model_name,
            completed=False
        )
