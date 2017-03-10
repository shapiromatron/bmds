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

from .session import BMDS
from .models.base import BMDModel
from .exceptions import RemoteBMDSExcecutionException

__all__ = []


def _get_payload(models):
    return dict(
        inputs=json.dumps([
            dict(
                bmds_version=model.bmds_version_dir,
                model_name=model.model_name,
                dfile=model.as_dfile(),
            ) for model in models
        ])
    )


if platform.system() != 'Windows':

    host = os.environ.get('BMDS_HOST')
    user = os.environ.get('BMDS_USERNAME')
    pw = os.environ.get('BMDS_PASSWORD')
    _request_session = None
    NO_HOST_WARNING = (
        'Using a non-Windows platform; BMDS cannot run natively in this OS.\n'
        'We can make a call to a remote server to execute.\n'
        'To execute BMDS, please specify the following environment variables:\n'
        '  - BMDS_HOST (e.g. http://bmds-server.com)\n'
        '  - BMDS_USERNAME (e.g. myusername)\n'
        '  - BMDS_PASSWORD (e.g. mysecret)\n'
    )

    def _get_requests_session():
        if host is None or user is None or pw is None:
            raise RemoteBMDSExcecutionException(NO_HOST_WARNING)

        global _request_session
        if _request_session is None:
            s = requests.Session()
            s.get('{}/admin/login/'.format(host))
            csrftoken = s.cookies['csrftoken']
            s.post('{}/admin/login/'.format(host), {
                'username': user,
                'password': pw,
                'csrfmiddlewaretoken': csrftoken,
            })

            # ensure authentication was successful
            if s.cookies.get('sessionid') is None:
                raise RemoteBMDSExcecutionException('Authentication failed')

            _request_session = s

        return _request_session

    def _set_outputs(model, result):
        model.output_created = result['output_created']
        if model.output_created:
            model.parse_results(result['outfile'])

    def execute_model(self):
        # execute single model
        if self.can_be_executed:
            session = _get_requests_session()
            url = '{}/dfile/'.format(host)
            payload = _get_payload([self])
            resp = session.post(url, data=payload)
            result = resp.json()[0]
        else:
            result = {'output_created': False}

        _set_outputs(self, result)

    def execute_session(self):
        # submit data
        executable_models = []
        for model in self.models:
            if model.can_be_executed:
                executable_models.append(model)
            else:
                _set_outputs(model, {'output_created': False})

        if len(executable_models) == 0:
            return

        session = _get_requests_session()
        url = '{}/dfile/'.format(host)
        payload = _get_payload(executable_models)
        resp = session.post(url, data=payload)

        # parse results for each model
        jsoned = resp.json()
        for model, result in zip(executable_models, jsoned):
            _set_outputs(model, result)

    # print startup error if host is None
    if host is None or user is None or pw is None:
        sys.stderr.write(NO_HOST_WARNING)

    # monkeypatch
    BMDS.execute = execute_session
    BMDModel.execute = execute_model
