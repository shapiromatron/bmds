import json
import os
import sys
from datetime import datetime

import bmds

HELP_TEXT = """BMDS multiprocessor execution:
------------------------------
    input_filename:     input file path
    output_filename:    output file path
"""


def create_datasets(fn):
    # parse an input file and create bmds.Dataset objects
    # returns a list of dataset objects
    fn = os.path.abspath(fn)
    with open(fn, "r") as f:
        raw_datasets = json.load(f)

    datasets = []
    for raw_dataset in raw_datasets["datasets"]:
        dataset = bmds.ContinuousIndividualDataset(**raw_dataset)
        datasets.append(dataset)

    return datasets


def execute(idx, dataset):
    session = bmds.BMDS.latest_version(bmds.constants.CONTINUOUS_INDIVIDUAL, dataset=dataset)
    session.add_default_models()
    try:
        session.execute()
        session.recommend()
    except Exception as e:
        print("Exception: {} {}".format(idx, dataset.kwargs.get("id")))
        raise e
    return session


def export(sessions, fn):
    # add sessions to a bmds.SessionBatch object; return JSON and excel outputs
    batch = bmds.SessionBatch()
    batch.extend(sessions)
    batch.to_json(fn, indent=2)
    batch.to_excel(fn + ".xlsx")


if __name__ == "__main__":

    # check that we're getting the correct input arguments
    if len(sys.argv) != 3:
        print(HELP_TEXT)
        quit()

    # expand whatever paths user provides to full-paths
    inputfn = os.path.abspath(os.path.expanduser(sys.argv[1]))
    outputfn = os.path.abspath(os.path.expanduser(sys.argv[2]))

    # create input datasets
    datasets = create_datasets(inputfn)

    # execute
    start = datetime.now()
    results = [execute(idx, dataset) for idx, dataset in enumerate(datasets)]
    end = datetime.now()

    # print some runtime benchmarks
    total_seconds = (end - start).total_seconds()
    total_minutes = total_seconds / 60
    print("Runtime: {:.2f} min for {} datasets".format(total_minutes, len(datasets)))
    print("{:.2f} seconds per dataset".format(total_seconds / len(datasets)))

    # export session results
    export(results, outputfn)
