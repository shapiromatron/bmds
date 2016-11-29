"""
If we're not working on Windows, we cannot execute BMDS.

The BMDS source-code can be for other operating systems, but the compiled
binaries can yield different results from the same inputs. Thus, we treat the
Windows binaries as the "standard" and therefore we don't attempt to use
the alternative compilations.

Instead, we monkeypatch the executable steps in running a model and a session.
The dfiles are passed via HTTP to a remote Windows server which will execute
the BMDS and return the result .OUT files in a JSON format.

This requires an additional environment variable, BMDS_HOST, which is the host
path for remote execution: e.g., the string "http://12.13.145.167".
"""

import json
import os
import platform
import requests
import sys

from .session import Session
from .models.base import BMDModel


if platform.system() != 'Windows':

    host = os.environ.get('BMDS_HOST', None)
    _request_session = None
    NO_HOST_WARNING = (
        'Using a non-Windows platform; BMDS cannot run natively in this OS.\n'
        'To execute BMDS, please specify a BMDS_HOST environment variable,\n'
        'and set this to a valid BMDS server root URL.\n'
    )

    def get_session():
        if host is None:
            raise EnvironmentError(NO_HOST_WARNING)

        global _request_session
        if _request_session is None:
            _request_session = requests.Session()
            # todo - authenticate here.
        return _request_session

    def get_payload(models):
        return dict(
            inputs=[
                dict(
                    bmds_version=model.bmds_version_dir,
                    model_app_name=model._get_model_name(),
                    dfile=model.as_dfile(),
                ) for model in models
            ]
        )

    def _set_outputs(model, result):
        model.output_created = result[0]
        if model.output_created:
            model.parse_results(result[1])

    def execute_model(self):
        # execute single model
        session = get_session()
        url = '{}/dfile/'.format(host)
        dataset = json.dumps(get_payload([self]))
        resp = session.post(url, dataset)

        # parse outputs
        result = resp.json()[0]
        _set_outputs(self, result)

    def execute_session(self):
        # submit data
        session = get_session()
        url = '{}/dfile/'.format(host)
        dataset = json.dumps(get_payload(self._models))
        resp = session.post(url, dataset)

        # parse results for each model
        jsoned = resp.json()
        for model, result in zip(self._models, jsoned):
            _set_outputs(model, result)

    # print startup error if host is None
    if host is None:
        sys.stderr.write(NO_HOST_WARNING)

    # monkeypatch
    Session.execute = execute_session
    BMDModel.execute = execute_model
