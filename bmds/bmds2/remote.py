"""
If we're not working on Windows, we cannot execute BMDS < 3.

The BMDS source-code can be for other operating systems, but the compiled
binaries can yield different results from the same inputs. Thus, we treat the
Windows binaries as the "standard" and therefore we don't attempt to use
the alternative compilations.

Instead, we monkeypatch the executable steps in running a model and a session.
The dfiles are passed via HTTP to a remote Windows server which will execute
the BMDS and return the result .OUT files in a JSON format.

We use a https://bmds-server.readthedocs.io/en/master/.
"""

import json
import logging
import platform
from datetime import datetime

import requests
from simple_settings import settings

from ..exceptions import RemoteBMDSExecutionException
from .models.base import BMDModel, RunStatus

logger = logging.getLogger(__name__)
__all__ = []


_request_session = None
NO_HOST_WARNING = (
    "Using a non-Windows platform; BMDS cannot run natively in this OS.\n"
    "We can make a call to a remote server to execute.\n"
    "To execute BMDS, please specify the following environment variables:\n"
    "  - BMDS_REQUEST_URL (e.g. http://bmds-server.com/api/dfile/)\n"
    "  - BMDS_TOKEN (e.g. 250b3c9cbcf448969a634400957e2c0849354d0d)"
)


def _get_payload(models):
    return json.dumps(
        dict(
            inputs=[
                dict(
                    bmds_version=model.bmds_version_dir,
                    model_name=model.model_name,
                    dfile=model.as_dfile(),
                )
                for model in models
            ]
        )
    )


def _get_requests_session():
    if settings.BMDS_REQUEST_URL is None or settings.BMDS_TOKEN is None:
        raise RemoteBMDSExecutionException(NO_HOST_WARNING)

    global _request_session
    if _request_session is None:
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Token {settings.BMDS_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        s._BMDS_REQUEST_URL = settings.BMDS_REQUEST_URL
        _request_session = s

    return _request_session


def _set_results(model, results=None):
    if results is None:
        model._set_job_outputs(RunStatus.DID_NOT_RUN)
    else:
        status = results.pop("status")
        if status == RunStatus.SUCCESS:
            model._set_job_outputs(RunStatus.SUCCESS, **results)
        elif status == RunStatus.FAILURE:
            model._set_job_outputs(RunStatus.FAILURE)


def execute_model(self):
    # execute single model
    self.execution_start = datetime.now()
    if self.can_be_executed:
        session = _get_requests_session()
        payload = _get_payload([self])
        logger.debug(f"Submitting payload: {payload}")
        resp = session.post(session._BMDS_REQUEST_URL, data=payload)
        results = resp.json()[0]
    else:
        results = None
    _set_results(self, results)


if platform.system() != "Windows":
    # monkeypatch
    BMDModel.execute = execute_model
