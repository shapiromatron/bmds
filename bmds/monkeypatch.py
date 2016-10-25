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

import random
import json
import os
import platform
import requests
import sys

from .session import Session
from .models.base import BMDModel


def get_dataset(models):
    return dict(
        options=dict(
            bmds_version=models[0].bmds_version_dir,
            emf_YN=False
        ),
        runs=[
            dict(
                id=random.randint(1, 1e10),
                model_app_name=model._get_model_name(),
                dfile=model.as_dfile()
            ) for model in models
        ]
    )

if platform.system() != 'Windows':
    NO_HOST_WARNING = (
        'Using a non-Windows platform; BMDS cannot run natively in this OS.\n'
        'To execute BMDS, please specify a BMDS_HOST environment variable,\n'
        'and set this to a valid BMDS server root URL.\n'
    )
    host = os.environ.get('BMDS_HOST', None)
    if host is None:
        sys.stderr.write(NO_HOST_WARNING)

    def execute_model(self):

        if host is None:
            raise EnvironmentError(NO_HOST_WARNING)

        # submit data
        url = '{}/Server-receiving.php'.format(host)
        dataset = json.dumps(get_dataset([self]))
        r = requests.post(url, dataset)
        job_id = r.json()['BMDS_Service_Number']

        # execute run
        url = '{}/BMDS_execution.php'.format(host)
        data = {'bsn': job_id}
        requests.get(url, data)

        # get results
        url = '{}/Server-report.php'.format(host)
        data = {'BMDS_Service_Number': job_id}
        r = requests.get(url, data)

        # parse results for each model
        resp = r.json()['BMDS_Results'][0]
        self.output_created = True
        self.parse_results(resp['OUT_file_str'])

    def execute_session(self):

        if host is None:
            raise EnvironmentError(NO_HOST_WARNING)

        # submit data
        url = '{}/Server-receiving.php'.format(host)
        dataset = json.dumps(get_dataset(self._models))
        r = requests.post(url, dataset)
        job_id = r.json()['BMDS_Service_Number']

        # execute run
        url = '{}/BMDS_execution.php'.format(host)
        data = {'bsn': job_id}
        requests.get(url, data)

        # get results
        url = '{}/Server-report.php'.format(host)
        data = {'BMDS_Service_Number': job_id}
        resp = requests.get(url, data)

        # parse results for each model
        response = resp.json()
        for model, result in zip(self._models, response['BMDS_Results']):
            model.output_created = True
            model.parse_results(result['OUT_file_str'])

    Session.execute = execute_session
    BMDModel.execute = execute_model
