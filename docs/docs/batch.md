# Batch Execution

hello


```python
import bmds
from bmds.bmds3.batch import BmdsSessionBatch, ExecutionResponse
from bmds.bmds3.sessions import Bmds330
from bmds.datasets import DichotomousDataset

# create multiple datasets
datasets = [
    DichotomousDataset(doses=[0, 10, 20, 30], ns=[10, 10, 10, 10], incidences=[0, 1, 2, 3]),
    DichotomousDataset(doses=[0, 10, 20, 30], ns=[10, 10, 10, 10], incidences=[0, 4, 5, 6])
]

# create a function that takes a dataset as input and returns a session response as output
def runner(ds):
    sess = Bmds330(dataset=ds)
    sess.add_model(bmds.constants.M_Logistic, settings={"bmr": 0.2})
    sess.execute_and_recommend()
    return ExecutionResponse(success=True, content=[sess.to_dict()])

# execute all datasets and sessions on a single processor
batch = BmdsSessionBatch().execute(datasets, runner, nprocs=1)

# save Excel and Word reports
batch.to_excel("report.xlsx")
batch.to_docx().save("report.docx")
```
