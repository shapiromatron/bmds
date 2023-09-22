# Run a Nested Dichotomous Dataset

## Quickstart

To run a nested dichotomous dataset:

```python
import bmds
from bmds import NestedDichotomousDataset
from bmds.bmds3.types.dichotomous import NestedDichotomousRiskType

dataset = NestedDichotomousDataset(name="Nested Dataset",
    dose_units="ppm",
    doses= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            25, 25, 25, 25, 25, 25, 25, 25, 25,
            50, 50, 50, 50, 50, 50, 50, 50, 50],
    litter_ns = [16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
            	9, 14, 9, 13, 12, 10, 10, 11, 14,
                11, 11, 14, 11, 10, 11, 10, 15, 7],
    incidences = [1, 1, 2, 3, 3, 0, 2, 2, 1, 2, 4,
                 5, 6, 2, 6, 3, 1, 2, 4, 3,
                 4, 5, 5, 4, 5, 4, 5, 6, 2],
    litter_covariates = [16, 9, 15, 14, 13, 9, 10, 14, 10, 11, 14,
                		9, 14, 9, 13, 12, 10, 10, 11, 14,
                		11, 11, 14, 11, 10, 11, 10, 15, 7]
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