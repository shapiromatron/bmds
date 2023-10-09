# Customize an Excel Export

One common request might be after executing a batch analysis, how to add some additional information to the default Excel exports. The `pybmds` package stores all modeling information in a data structure that allows us to get both the data and reports. This notebook demonstrates running a simple batch analysis, and then augmenting the default Excel export with a few extra columns of information.

```python
import json
from pathlib import Path
from pprint import pprint

import pandas as pd

import bmds
from bmds import constants
from bmds.bmds3.batch import BmdsSessionBatch
from bmds.bmds3.batch import ExecutionResponse
from bmds.bmds3.constants import DistType, PriorClass
from bmds.bmds3.sessions import Bmds330
from bmds.bmds3.types.continuous import ContinuousRiskType
from bmds.bmds3.types.dichotomous import DichotomousRiskType
from bmds.datasets import DichotomousDataset, ContinuousDataset
```

As a simple example, we'll generate a batch analysis using a few option sets and a single dataset. You could adapt this code to run a custom analysis of your choosing:

```python
def build_cont_sess(ds):
    def add_model(sess, Model, base, additions=None):
        settings = {
            "priors": PriorClass.frequentist_restricted,
            "bmr_type": base[0],
            "bmr": base[1],
            "disttype": base[2],
        }
        if additions is not None:
            settings.update(additions)
        sess.add_model(Model, settings)

    option_sets = [
        (ContinuousRiskType.RelativeDeviation, 0.1, DistType.normal),
        (ContinuousRiskType.RelativeDeviation, 0.1, DistType.normal_ncv),
    ]
    sessions = []
    for option_set in option_sets:
        sess = Bmds330(dataset=ds)
        add_model(sess, constants.M_ExponentialM3, option_set)
        add_model(sess, constants.M_ExponentialM5, option_set)
        add_model(sess, constants.M_Power, option_set)
        add_model(sess, constants.M_Power, option_set)
        add_model(sess, constants.M_Linear, option_set)
        add_model(sess, constants.M_Polynomial, option_set, {"degree": 2})
        add_model(sess, constants.M_Polynomial, option_set, {"degree": 3})        
        sess.execute_and_recommend()
        sessions.append(sess.to_dict())
    return ExecutionResponse(success=True, content=sessions)


datasets = [
    bmds.ContinuousDataset(
        doses=[0, 10, 50, 150, 400],
        ns=[111, 142, 143, 93, 42],
        means=[2.112, 2.095, 1.956, 1.587, 1.254],
        stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
    )        
]
sess_batch = BmdsSessionBatch.execute(datasets, build_cont_sess, nprocs=1)
```

To generate a standard Excel export, you'd call this method:

`sess_batch.to_excel('./filename.xlsx')`

However, we'll want to customize this export to add more information. In this example, we may want to show more information regarding Analysis of Deviance Table than is generally shown in the summary exports. First, let's generate the the summary dataframe, which we'll want to add more info to:

```python
df_summary = sess_batch.df_summary()
df_summary.head()
```
Which has 5 rows and 35 columns. Now, let's add some additional information to the summary table. We'll iterate all the sessions in our dataset, and all the models in each session, and create a new data frame that we can merge with the default summary data frame:

```python
rows = []
for i, session in enumerate(sess_batch.sessions):
    for j, model in enumerate(session.models):
        data = {
            "session_index": i,            
            "model_index": j,            
        }
        if model.has_results:
            res = model.results
            data.update({
                "deviance": res.deviance.tbl(),
                "A1_ll": res.deviance.loglikelihoods[0],
                "A2_ll": res.deviance.loglikelihoods[1],
                "A3_ll": res.deviance.loglikelihoods[2],
                "fitted_ll": res.deviance.loglikelihoods[3],
                "reduced_ll": res.deviance.loglikelihoods[4],
                "A1_aic": res.deviance.aics[0],
                "A2_aic": res.deviance.aics[1],
                "A3_aic": res.deviance.aics[2],
                "fitted_aic": res.deviance.aics[3],
                "reduced_aic": res.deviance.aics[4],
            })
        rows.append(data)
        
df2 = pd.DataFrame(rows)
df2.head()
```

Now, we can join the two data frames together, using the session and model keys to join:

```python
df_summary2 = pd.merge(df_summary, df2, on=["session_index", "model_index"])
df_summary2.head()
```

Now, the summary dataframe has 46 columns instead of 35.

Finally, we'll write the Excel export:

```python
report_filename = "custom.xlsx"

with pd.ExcelWriter(report_filename) as writer:
    data = {
        "summary": df_summary2,
        "datasets": sess_batch.df_dataset(),
        "parameters": sess_batch.df_params(),
    }
    for name, df in data.items():
        df.to_excel(writer, sheet_name=name, index=False)
```

This export includes our custom values!

## Introspecting Model Results

A quick dive into introspecting the model results that are available and their data structures and text summaries.

Let's grab the first model that was executed and look at it's results object:

```python
model = sess_batch.sessions[0].models[0]
res = model.results
```

This is a nested python data structure, for example:

```python
print(f"{res.bmd=}")
print(f"{res.bmdl=}")
print(f"{res.fit.aic=}")

res.bmd=69.7195053100586
res.bmdl=64.711210521579
res.fit.aic=-40.00553974137064
```

To better understand the structure, we can "pretty print" a dictionary representation:

```python
data = res.dict()

# truncate a few fields so they print better... (you can ignore this code)
data['fit']['bmd_dist'][0] = data['fit']['bmd_dist'][0][:5]
data['fit']['bmd_dist'][1] = data['fit']['bmd_dist'][1][:5]
data['plotting']['dr_x'] = data['plotting']['dr_x'][:5]
data['plotting']['dr_y'] = data['plotting']['dr_y'][:5]

pprint(data, indent=2, width=140, sort_dicts=True)

{ 'bmd': 69.7195053100586,
  'bmdl': 64.711210521579,
  'bmdu': 75.42553250760992,
  'deviance': { 'aics': [-58.761008514924924, -66.16149887860908, -58.761008514924924, -40.00553974137064, 392.9064834355403],
                'loglikelihoods': [35.38050425746246, 43.08074943930454, 35.38050425746246, 23.00276987068532, -194.45324171777014],
                'names': ['A1', 'A2', 'A3', 'fitted', 'reduced'],
                'num_params': [6, 10, 6, 3, 2]},
  'fit': { 'aic': -40.00553974137064,
           'bic_equiv': 48.898730820089426,
           'bmd_dist': [ [64.711210521579, 64.90918259298849, 65.11054834577763, 65.31288434006723, 65.51376713597806],
                         [0.05, 0.06, 0.07, 0.08, 0.09]],
           'chisq': -9999.0,
           'dist': 1,
           'loglikelihood': -23.00276987068532,
           'model_df': 4.0,
           'total_df': 1.0},
  'gof': { 'calc_mean': [2.112, 2.095, 1.956, 1.587, 1.254],
           'calc_sd': [0.235, 0.209, 0.231, 0.263, 0.159],
           'dose': [0.0, 10.0, 50.0, 150.0, 400.0],
           'eb_lower': [2.05007080082289, 2.046304233630524, 1.9023668811812215, 1.5112812622766696, 1.1858820498658582],
           'eb_upper': [2.1739291991771097, 2.143695766369476, 2.0096331188187784, 1.6627187377233301, 1.3221179501341416],
           'est_mean': [2.1049649432922055, 2.07339374237006, 1.9517736384124835, 1.6780257038263129, 1.1500626830207765],
           'est_sd': [0.2317124187140318, 0.2317124187140318, 0.2317124187140318, 0.2317124187140318, 0.2317124187140318],
           'obs_mean': [2.112, 2.095, 1.956, 1.587, 1.254],
           'obs_sd': [0.235, 0.209, 0.231, 0.263, 0.159],
           'residual': [0.3198746187897758, 1.1111544038616936, 0.21811491217707857, -3.788403327047847, 2.907012079753416],
           'roi': 0.21811491217707857,
           'size': [111.0, 142.0, 143.0, 93.0, 42.0]},
  'has_completed': True,
  'parameters': { 'bounded': [0.0, 0.0, 1.0, 0.0],
                  'cov': [ [0.00017089187214269089, 4.899375110470095e-07, -1.5251765872285089e-12, 1.202387298918987e-10],
                           [4.899375110470096e-07, 4.820960799209715e-09, 9.970972457337382e-15, 1.067506519904174e-12],
                           [-1.5251765872285087e-12, 9.970972457337384e-15, 9.378157620616477e-12, 1.7225834792422805e-12],
                           [1.2023872989189873e-10, 1.0675065199041742e-12, 1.7225834792422805e-12, 0.003766474079956207]],
                  'lower_ci': [2.0793431927668444, 0.0013751194598262198, -9999.0, -3.0448026407558935],
                  'names': ['a', 'b', 'd', 'log-alpha'],
                  'prior_initial_value': [0.0, 0.0, 0.0, 0.0],
                  'prior_max_value': [100.0, 100.0, 18.0, 18.0],
                  'prior_min_value': [0.0, 0.0, 1.0, -18.0],
                  'prior_stdev': [0.0, 0.0, 0.0, 0.0],
                  'prior_type': [0, 0, 0, 0],
                  'se': [0.013072561804890841, 6.94331390562872e-05, -9999.0, 0.06137160646387063],
                  'upper_ci': [2.1305866938175666, 0.0016472923657408537, -9999.0, -2.804230362173186],
                  'values': [2.1049649432922055, 0.0015112059127835367, 1.0, -2.92451650146454]},
  'plotting': { 'bmd_y': 1.8944684243290653,
                'bmdl_y': 1.9088612288964968,
                'bmdu_y': 1.8782026874976185,
                'dr_x': [1e-08, 4.040404040404041, 8.080808080808081, 12.121212121212121, 16.161616161616163],
                'dr_y': [2.1049649432603954, 2.092151433434438, 2.0794159229920046, 2.0667579371606664, 2.054177004026451]},
  'tests': { 'dfs': [8.0, 4.0, 4.0, 3.0],
             'll_ratios': [475.0679823141494, 15.400490363684156, 15.400490363684156, 24.755468773554284],
             'names': ['Test 1', 'Test 2', 'Test 3', 'Test 4'],
             'p_values': [0.0, 0.003938741689778258, 0.003938741689778258, 1.736920733863556e-05]}}
```

This may be helpful in trying to find particular values:

`print(res.fit.loglikelihood)
-23.00276987068532`

This is a helpful pattern to print data in a tabular format:

```python
for name, df, ll, p_value in zip(res.tests.names, res.tests.dfs, res.tests.ll_ratios, res.tests.p_values, strict=True):
    print(f"{name:10} {df: <6} {ll: <10.4f} {p_value: <8.6f}")
```

## Text Reports

In addition to raw data, a text-based report can be generated for each model:

```python
print(model.text())
   ExponentialM3    
════════════════════

Input Summary:
╒════════════════════╤════════════════════════════╕
│ BMR                │ 10% Relative Deviation     │
│ Distribution       │ Normal + Constant variance │
│ Modeling Direction │ Down (↓)                   │
│ Confidence Level   │ 0.95                       │
│ Tail Probability   │ 0.01                       │
│ Modeling Approach  │ Frequentist restricted     │
╘════════════════════╧════════════════════════════╛

Parameter Settings:
╒═════════════╤═══════════╤═══════╤═══════╕
│ Parameter   │   Initial │   Min │   Max │
╞═════════════╪═══════════╪═══════╪═══════╡
│ a           │         0 │     0 │   100 │
│ b           │         0 │     0 │   100 │
│ c           │         0 │   -20 │     0 │
│ d           │         0 │     1 │    18 │
│ log-alpha   │         0 │   -18 │    18 │
╘═════════════╧═══════════╧═══════╧═══════╛

Summary:
╒════════════════╤═══════════════╕
│ BMD            │  69.7195      │
│ BMDL           │  64.7112      │
│ BMDU           │  75.4255      │
│ AIC            │ -40.0055      │
│ Log Likelihood │ -23.0028      │
│ P-Value        │   1.73692e-05 │
│ Model DOF      │   3           │
╘════════════════╧═══════════════╛

Model Parameters:
╒════════════╤═════════════╤═══════════╤═════════════╤════════════╤════════════╕
│ Variable   │    Estimate │ Bounded   │ Std Error   │ Lower CI   │ Upper CI   │
╞════════════╪═════════════╪═══════════╪═════════════╪════════════╪════════════╡
│ a          │  2.10496    │ no        │ 0.0130726   │ 2.07934    │ 2.13059    │
│ b          │  0.00151121 │ no        │ 6.94331e-05 │ 0.00137512 │ 0.00164729 │
│ d          │  1          │ yes       │ NA          │ NA         │ NA         │
│ log-alpha  │ -2.92452    │ no        │ 0.0613716   │ -3.0448    │ -2.80423   │
╘════════════╧═════════════╧═══════════╧═════════════╧════════════╧════════════╛

Goodness of Fit:
╒════════╤════════╤═════════════════╤═══════════════════╤══════════════════╤═══════════════════╕
│   Dose │   Size │   Observed Mean │   Calculated Mean │   Estimated Mean │   Scaled Residual │
╞════════╪════════╪═════════════════╪═══════════════════╪══════════════════╪═══════════════════╡
│      0 │    111 │           2.112 │             2.112 │          2.10496 │          0.319875 │
│     10 │    142 │           2.095 │             2.095 │          2.07339 │          1.11115  │
│     50 │    143 │           1.956 │             1.956 │          1.95177 │          0.218115 │
│    150 │     93 │           1.587 │             1.587 │          1.67803 │         -3.7884   │
│    400 │     42 │           1.254 │             1.254 │          1.15006 │          2.90701  │
╘════════╧════════╧═════════════════╧═══════════════════╧══════════════════╧═══════════════════╛
╒════════╤════════╤═══════════════╤═════════════════╤════════════════╕
│   Dose │   Size │   Observed SD │   Calculated SD │   Estimated SD │
╞════════╪════════╪═══════════════╪═════════════════╪════════════════╡
│      0 │    111 │         0.235 │           0.235 │       0.231712 │
│     10 │    142 │         0.209 │           0.209 │       0.231712 │
│     50 │    143 │         0.231 │           0.231 │       0.231712 │
│    150 │     93 │         0.263 │           0.263 │       0.231712 │
│    400 │     42 │         0.159 │           0.159 │       0.231712 │
╘════════╧════════╧═══════════════╧═════════════════╧════════════════╛

Likelihoods of Interest:
╒═════════╤══════════════════╤════════════╤══════════╕
│ Model   │   Log Likelihood │   # Params │      AIC │
╞═════════╪══════════════════╪════════════╪══════════╡
│ A1      │          35.3805 │          6 │ -58.761  │
│ A2      │          43.0807 │         10 │ -66.1615 │
│ A3      │          35.3805 │          6 │ -58.761  │
│ fitted  │          23.0028 │          3 │ -40.0055 │
│ reduced │        -194.453  │          2 │ 392.906  │
╘═════════╧══════════════════╧════════════╧══════════╛

Tests of Interest:
╒════════╤═══════════════════════╤════════════╤═════════════╕
│ Name   │   Loglikelihood Ratio │   Test DOF │     P-Value │
╞════════╪═══════════════════════╪════════════╪═════════════╡
│ Test 1 │              475.068  │          8 │ 0           │
│ Test 2 │               15.4005 │          4 │ 0.00393874  │
│ Test 3 │               15.4005 │          4 │ 0.00393874  │
│ Test 4 │               24.7555 │          3 │ 1.73692e-05 │
╘════════╧═══════════════════════╧════════════╧═════════════╛
```

Individual table components can also be generated:

```python
print(res.parameters.tbl())
╒════════════╤═════════════╤═══════════╤═════════════╤════════════╤════════════╕
│ Variable   │    Estimate │ Bounded   │ Std Error   │ Lower CI   │ Upper CI   │
╞════════════╪═════════════╪═══════════╪═════════════╪════════════╪════════════╡
│ a          │  2.10496    │ no        │ 0.0130726   │ 2.07934    │ 2.13059    │
│ b          │  0.00151121 │ no        │ 6.94331e-05 │ 0.00137512 │ 0.00164729 │
│ d          │  1          │ yes       │ NA          │ NA         │ NA         │
│ log-alpha  │ -2.92452    │ no        │ 0.0613716   │ -3.0448    │ -2.80423   │
╘════════════╧═════════════╧═══════════╧═════════════╧════════════╧════════════╛

print(res.deviance.tbl())
╒═════════╤══════════════════╤════════════╤══════════╕
│ Model   │   Log Likelihood │   # Params │      AIC │
╞═════════╪══════════════════╪════════════╪══════════╡
│ A1      │          35.3805 │          6 │ -58.761  │
│ A2      │          43.0807 │         10 │ -66.1615 │
│ A3      │          35.3805 │          6 │ -58.761  │
│ fitted  │          23.0028 │          3 │ -40.0055 │
│ reduced │        -194.453  │          2 │ 392.906  │
╘═════════╧══════════════════╧════════════╧══════════╛
print(res.tests.tbl())
╒════════╤═══════════════════════╤════════════╤═════════════╕
│ Name   │   Loglikelihood Ratio │   Test DOF │     P-Value │
╞════════╪═══════════════════════╪════════════╪═════════════╡
│ Test 1 │              475.068  │          8 │ 0           │
│ Test 2 │               15.4005 │          4 │ 0.00393874  │
│ Test 3 │               15.4005 │          4 │ 0.00393874  │
│ Test 4 │               24.7555 │          3 │ 1.73692e-05 │
╘════════╧═══════════════════════╧════════════╧═════════════╛
```