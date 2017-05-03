from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import sys

import bmds


HELP_TEXT = '''BMDS multiprocessor execution:
------------------------------
    input_filename:     input file path
    output_filename:    output file path
'''


def create_datasets(fn):
    # parse an input file and create bmds.Dataset objects
    # returns a list of dataset objects
    fn = os.path.abspath(fn)
    with open(fn, 'r') as f:
        raw_datasets = json.load(f)

    datasets = []
    for raw_dataset in raw_datasets:
        dataset = bmds.ContinuousDataset(**raw_dataset)
        datasets.append(dataset)

    return datasets


def execute(dataset):
    # configure BMDS session and execute
    session = bmds.BMDS.latest_version(
        bmds.constants.CONTINUOUS,
        dataset=dataset)
    session.add_default_models()
    session.execute()
    session.recommend()
    return session


def export_as_json(fn, sessions):
    # add sessions to a bmds.SessionBatch object and return JSON output
    batch = bmds.SessionBatch()
    batch.extend(sessions)
    batch.to_json(fn, indent=2)


# must start multiprocessor in main-body in Windows
# https://docs.python.org/3/library/multiprocessing.html
if __name__ == '__main__':

    # check that we're getting the correct input arguments
    if len(sys.argv) != 3:
        print(HELP_TEXT)
        quit()

    # expand whatever paths user provides to full-paths
    inputfn = os.path.abspath(os.path.expanduser(sys.argv[1]))
    outputfn = os.path.abspath(os.path.expanduser(sys.argv[2]))

    # create input datasets
    datasets = create_datasets(inputfn)

    # use n-1 processors for execution
    nprocs = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        results = executor.map(execute, datasets)

    # export all sessions as JSON
    export_as_json(outputfn, list(results))
