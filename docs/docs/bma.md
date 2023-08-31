# Run Bayesian Model Averaging for a Dichotomous Dataset

Bayesian model averaging is currently only available for dichotomous datasets in BMDS. Here, we will go over how to run a dichotomous analysis with Bayesian model averaging and how to plot your results. Also, we will demonstrate how you can override the default priors for parameter estimation. 

## Quickstart

To run Bayesian model averaging:

```python
import bmds
from bmds import DichotomousDataset
from bmds.bmds3.models import dichotomous
from bmds.bmds3.types.dichotomous import DichotomousRiskType

# create a dichotomous dataset
dataset = DichotomousDataset(
    doses=[0, 25, 75, 125, 200],
    ns=[20, 20, 20, 20, 20],
    incidences=[0, 1, 7, 15, 19],
)

session1 = bmds.BMDS.latest_version(dataset=dataset)
session1.add_default_bayesian_models()
session1.execute()

res = session1.model_average.results
print(f"BMD = {res.bmd:.2f} [{res.bmdl:.2f}, {res.bmdu:.2f}]")

bma_plot = session1.model_average.plot()
bma_plot.savefig("bma.png")
```

Here, you will get an output of the BMD [BMDL, BMDU], as:

```python
BMD = 36.65 [17.81, 53.97]
```

and a plot that will show the model average fit:
![image info](bma.png)

## Changing the input settings

The default settings for a dichotomous Bayesian model averaged run use a BMR of 10% Extra Risk and a 95% confidence interval. You can change these settings by:

```python
session1.add_default_bayesian_models(global_settings = {"bmr": 0.05, "bmr_type": DichotomousRiskType.AddedRisk, "alpha": 0.1})
```

This would run the dichotomous models for a BMR of 5% Added Risk at a 90% confidence interval.