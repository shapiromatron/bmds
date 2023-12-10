import json

from bmds.bmds3.constants import PriorClass, PriorType
from bmds.bmds3.models.dichotomous import Multistage
from bmds.bmds3.models.multi_tumor import MultistageCancer, Multitumor
from bmds.bmds3.types.dichotomous import DichotomousModelSettings
from bmds.bmds3.types.priors import ModelPriors, Prior
from bmds.datasets import DichotomousDataset


class TestMultitumor:
    def test_execute(self, rewrite_data_files, data_path):
        ds1 = DichotomousDataset(
            doses=[0, 50, 100, 150, 200],
            ns=[100, 100, 100, 100, 100],
            incidences=[0, 5, 30, 65, 90],
        )
        ds2 = DichotomousDataset(
            doses=[0, 50, 100, 150, 200],
            ns=[100, 100, 100, 100, 100],
            incidences=[5, 10, 33, 67, 93],
        )
        ds3 = DichotomousDataset(
            doses=[0, 50, 100, 150, 200],
            ns=[100, 100, 100, 100, 100],
            incidences=[1, 68, 78, 88, 98],
        )
        datasets = [ds1, ds2, ds3]
        degrees = [3, 0, 0]
        session = Multitumor(datasets, degrees=degrees, id=1, name="test", description="hello")
        session.execute()

        # check text report
        text = session.text()
        assert len(text) > 0

        # check serialization
        session2 = session.serialize().deserialize()
        assert session.to_dict() == session2.to_dict()

        # check that individual models has slope factor and shown in text output
        assert session.models[0][0].results.slope_factor > 0
        assert "Slope Factor" in session.models[0][0].text()

        # dataframe
        df = session.to_df()
        assert "slope_factor" in df.columns

        # docx
        docx = session.to_docx()

        if rewrite_data_files:
            (data_path / "bmds3-mt.txt").write_text(text)
            df.to_excel(data_path / "bmds3-mt.xlsx", index=False)
            docx.save(data_path / "bmds3-mt.docx")


class TestMultistageCancer:
    def test_settings(self, ddataset2):
        default = json.loads(
            '{"prior_class": 1, "priors": [{"name": "g", "type": 0, "initial_value": -17.0, "stdev": 0.0, "min_value": -18.0, "max_value": 18.0}, {"name": "b1", "type": 0, "initial_value": 0.1, "stdev": 0.0, "min_value": 0.0, "max_value": 10000.0}, {"name": "bN", "type": 0, "initial_value": 0.1, "stdev": 0.0, "min_value": 0.0, "max_value": 10000.0}], "variance_priors": null, "beta_overrides": null}'
        )

        # default MultistageCancer use cancer prior
        model = MultistageCancer(ddataset2)
        assert model.settings.bmr == 0.1
        assert model.settings.priors.model_dump() == default

        # MultistageCancer w/ custom settings, but unspecified prior use cancer prior
        model = MultistageCancer(ddataset2, settings=dict(bmr=0.2))
        assert model.settings.bmr == 0.2
        assert model.settings.priors.model_dump() == default

        # MultistageCancer w/ DichotomousModelSettings is preserved
        base_model = Multistage(ddataset2)
        model = MultistageCancer(ddataset2, DichotomousModelSettings())
        assert model.settings.bmr == 0.1
        assert model.settings.priors.model_dump() != default
        assert model.settings.priors.model_dump() == base_model.settings.priors.model_dump()

        # MultistageCancer w/ custom settings is preserved
        custom = ModelPriors(
            prior_class=PriorClass.frequentist_restricted,
            priors=[
                Prior(
                    name="g",
                    type=PriorType.Uniform,
                    initial_value=0,
                    stdev=0,
                    min_value=-1,
                    max_value=1,
                )
            ],
            variance_priors=None,
        )
        model = MultistageCancer(ddataset2, settings=dict(bmr=0.2, priors=custom))
        assert model.settings.bmr == 0.2
        assert model.settings.priors.model_dump() == custom.model_dump()
