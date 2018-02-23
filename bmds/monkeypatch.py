"""
If we're not working on Windows, we cannot execute BMDS.

The BMDS source-code can be for other operating systems, but the compiled
binaries can yield different results from the same inputs. Thus, we treat the
Windows binaries as the "standard" and therefore we don't attempt to use
the alternative compilations.

Instead, we monkeypatch the executable steps in running a model and a session.
The dfiles are passed via HTTP to a remote Windows server which will execute
the BMDS and return the result .OUT files in a JSON format.

We use a https://bmds-server.readthedocs.io/en/master/.
"""

from datetime import datetime
import json
import logging
import platform
import requests
from simple_settings import settings

from .session import BMDS
from .models.base import BMDModel, RunStatus
from .exceptions import RemoteBMDSExcecutionException


logger = logging.getLogger(__name__)
__all__ = []


def _get_payload(models):
    return json.dumps(dict(inputs=[
        dict(
            bmds_version=model.bmds_version_dir,
            model_name=model.model_name,
            dfile=model.as_dfile(),
        ) for model in models]
    ))


if platform.system() != 'Windows':

    _request_session = None
    NO_HOST_WARNING = (
        'Using a non-Windows platform; BMDS cannot run natively in this OS.\n'
        'We can make a call to a remote server to execute.\n'
        'To execute BMDS, please specify the following environment variables:\n'
        '  - BMDS_REQUEST_URL (e.g. http://bmds-server.com/api/dfile/)\n'
        '  - BMDS_TOKEN (e.g. 250b3c9cbcf448969a634400957e2c0849354d0d)'
    )

    def _get_requests_session():
        if settings.BMDS_REQUEST_URL is None or \
           settings.BMDS_TOKEN is None:
                raise RemoteBMDSExcecutionException(NO_HOST_WARNING)

        global _request_session
        if _request_session is None:
            s = requests.Session()
            s.headers.update({
                'Authorization': 'Token {}'.format(settings.BMDS_TOKEN),
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            s._BMDS_REQUEST_URL = settings.BMDS_REQUEST_URL
            _request_session = s

        return _request_session

    def _set_results(model, results=None):
        if results is None:
            model._set_job_outputs(RunStatus.DID_NOT_RUN)
        else:
            status = results.pop('status')
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
            logger.debug('Submitting payload: {}'.format(payload))
            resp = session.post(session._BMDS_REQUEST_URL, data=payload)
            results = resp.json()[0]
        else:
            results = None
        _set_results(self, results)

    def execute_session(self):
        # submit data
        start_time = datetime.now()
        executable_models = []
        for model in self.models:
            model.execution_start = start_time
            if model.can_be_executed:
                executable_models.append(model)
            else:
                _set_results(model)

        if len(executable_models) == 0:
            return

        session = _get_requests_session()
        payload = _get_payload(executable_models)
        logger.debug('Submitting payload: {}'.format(payload))
        resp = session.post(session._BMDS_REQUEST_URL, data=payload)

        # parse results for each model
        jsoned = resp.json()
        for model, results in zip(executable_models, jsoned):
            _set_results(model, results)

    # monkeypatch
    BMDS.execute = execute_session
    BMDModel.execute = execute_model
