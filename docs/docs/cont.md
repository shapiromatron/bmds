# Run a Continuous Dataset 

To run a continuous dataset:

```python
import bmds
from bmds import ContinuousDataset
from bmds.bmds3.constants import DistType, PriorClass
from bmds.bmds3.types.continuous import ContinuousRiskType

# create a continuous dataset
dataset = ContinuousDataset(
    doses=[0, 25, 50, 75, 100],
    ns=[20, 20, 20, 20, 20],
    means=[6, 8, 13, 25, 30],
    stdevs=[4, 4.3, 3.8, 4.4, 3.7]
)

session = bmds.BMDS.latest_version(dataset=dataset)

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

The default settings when running a continuous analysis are a BMR of 1 standard deviation from the BMD, a 95% confidence interval, and a normally distributed and constant variance between dose groups. To adjust the settings you can:

```python
session.add_default_models(global_settings = {"disttype": DistType.normal_ncv, "bmr_type": ContinuousRiskType.AbsoluteDeviation, "bmr": 2, "alpha": 0.1})
```

Here, we changed the settings to assume normally distributed and non-constant variance between dose groups and a 90% confidence interval. We also changed the BMR type to be absolute deviation with a value of 2. You can use the default BMR for the given BMR type or specify a different value. You can also change the```"disttype"``` to be ```"disttype": DistType.log_normal```. If a log-normal distribution is selected, only the Hill and Exponential models will be run.

You can also change ```"bmr_type"``` several other ways. The options for ```"bmr_type"``` are:

```python
ContinuousRiskType.AbsoluteDeviation: "{} Absolute Deviation" 
ContinuousRiskType.StandardDeviation: "{} Standard Deviation" #default
ContinuousRiskType.RelativeDeviation: "{:.0%} Relative Deviation"
ContinuousRiskType.PointEstimate: "{} Point Estimation"
ContinuousRiskType.Extra: "{} Extra"
ContinuousRiskType.HybridExtra: "{} Hybrid Extra"
ContinuousRiskType.HybridAdded: "{} Hybrid Added"
```
 
