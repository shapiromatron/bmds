# Python BMDS

[![pypi](https://img.shields.io/pypi/v/bmds.svg)](https://pypi.python.org/pypi/bmds)
[![actions](https://github.com/shapiromatron/bmds/workflows/CI/badge.svg)](https://github.com/shapiromatron/bmds/actions)
[![docs](https://readthedocs.org/projects/bmds/badge/?version=latest)](https://bmds.readthedocs.io/en/latest/?badge=latest)
[![zenodo](https://zenodo.org/badge/61229626.svg)](https://zenodo.org/badge/latestdoi/61229626)

A python package is designed to run the [USEPA BMDS](https://www.epa.gov/bmds) software. It requires Python3.9+.

## Quickstart

Install the software using pip:

```bash
pip install bmds
```

An example dichotomous dataset:

```python
import bmds

# create a dataset
dataset = bmds.DichotomousDataset(
    doses=[0, 10, 50, 150, 400],
    ns=[25, 25, 24, 24, 24],
    incidences=[0, 3, 7, 11, 15],
)

# create a BMD session
session = bmds.BMDS.latest_version(dataset=dataset)

# add all default models
session.add_default_models()

# execute the session
session.execute()

# recommend a best-fitting model
session.recommend()

model_index = session.recommender.results.recommended_model_index
if model_index:
    model = session.models[model_index]
    print(model.text())

# save excel report
df = session.to_df()
df.to_excel("report.xlsx")

# save to a word report
report = session.to_docx()
report.save("report.docx")
```
