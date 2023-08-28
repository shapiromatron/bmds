# Run a Dichotomous Dataset

To run a dichotomous dataset:

```python
import bmds
from bmds import DichotomousDataset
from bmds.bmds3.types.dichotomous import DichotomousRiskType

dataset = DichotomousDataset(
    doses=[0, 25, 75, 125, 200],
    ns=[20, 20, 20, 20, 20],
    incidences=[0, 1, 7, 15, 19],
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

# plotting all models on one plot
def plot(colorize: bool = False):
   
    dataset = session.dataset
    results = session.execute()
    fig = dataset.plot()
    ax = fig.gca()
    ax.set_ylim(-0.05, 1.05)
    title = f"{dataset._get_dataset_name()}\nDichotomous Models, 10% Extra Risk" ## change title if you used a different BMR to calculate BMD
    ax.set_title(title)
    if colorize:
        color_cycle = cycle(plotting.INDIVIDUAL_MODEL_COLORS)
        line_cycle = cycle(plotting.INDIVIDUAL_LINE_STYLES)
    else:
        color_cycle = cycle(["#ababab"])
        line_cycle = cycle(["solid"])
    for i, model in enumerate(session.models):
        if colorize:
            label = model.name()
            print(label)
        elif i == 0:
            label = "Individual models"
        else:
            label = None
        ax.plot(
            model.results.plotting.dr_x,
            model.results.plotting.dr_y,
            label=label,
            c=next(color_cycle),
            linestyle=next(line_cycle),
            zorder=100,
            lw=2,
        )


    # reorder handles and labels
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
    )

    return fig


plot = plot(True)
plot.savefig("dichotomous-dataset.png")
```

The default settings for a dichotomous run use a BMR of 10% Extra Risk and a 95% confidence interval. You can change these settings by:

```python
session.add_default_models(global_settings = {"bmr": 0.15, "bmr_type": DichotomousRiskType.AddedRisk, "alpha": 0.1})
```

This would run the dichotomous models for a BMR of 15% Added Risk at a 90% confidence interval.
