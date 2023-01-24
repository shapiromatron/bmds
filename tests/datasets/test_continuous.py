from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

import bmds

from ..bmds2.run import windows_only

dummy2 = [1, 2]
dummy3 = [1, 2, 3]
dummy4 = [1, 2, 3, 4]
dummy3_dups = [0, 0, 1]
dummy3_floats = [0.1, 0.2, 0.3]


class TestContinuousSummaryDataset:
    def test_validation(self):
        # these should be valid
        bmds.ContinuousDataset(doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3)
        # some data adjustments result in non-integer based counts
        bmds.ContinuousDataset(doses=dummy3, ns=dummy3_floats, means=dummy3, stdevs=dummy3)
        # these should raise errors
        with pytest.raises((IndexError, ValueError)):
            # insufficient number of dose groups
            bmds.ContinuousDataset(doses=dummy2, ns=dummy2, means=dummy2, stdevs=dummy2)
            # different sized lists
            bmds.ContinuousDataset(doses=dummy4, ns=dummy3, means=dummy3, stdevs=dummy3)
            # zero in ns data
            bmds.ContinuousDataset(doses=dummy3, ns=[0, 2, 3], means=dummy3, stdevs=dummy3)

    def test_extra_kwargs(self):
        ds = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[111, 142, 143, 93, 42],
            means=[2.112, 2.095, 1.956, 1.587, 1.254],
            stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
        )
        assert ds.to_dict()["metadata"] == {
            "id": None,
            "name": "",
            "dose_units": "",
            "response_units": "",
            "dose_name": "",
            "response_name": "",
        }

        assert ds.get_xlabel() == "Dose"
        assert ds.get_ylabel() == "Response"

        ds = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[111, 142, 143, 93, 42],
            means=[2.112, 2.095, 1.956, 1.587, 1.254],
            stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
            id=123,
            name="example dataset",
            dose_units="mg/kg/d",
            response_units="ug/m3",
            dose_name="Intravenous",
            response_name="Volume",
        )
        assert ds.to_dict()["metadata"] == {
            "id": 123,
            "name": "example dataset",
            "dose_units": "mg/kg/d",
            "response_units": "ug/m3",
            "dose_name": "Intravenous",
            "response_name": "Volume",
        }

        assert ds.get_xlabel() == "Intravenous (mg/kg/d)"
        assert ds.get_ylabel() == "Volume (ug/m3)"

    def test_is_increasing(self):
        ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=dummy4, stdevs=dummy4)
        assert ds.is_increasing is True

        rev = list(reversed(dummy4))
        ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=rev, stdevs=dummy4)
        assert ds.is_increasing is False

        ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=[1, 2, 3, 0], stdevs=dummy4)
        assert ds.is_increasing is True

        ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=[1, 3, 2, 1], stdevs=dummy4)
        assert ds.is_increasing is True

        ds = bmds.ContinuousDataset(doses=dummy4, ns=dummy4, means=[0, 2, -1, 0], stdevs=dummy4)
        assert ds.is_increasing is True

    @windows_only
    def test_correct_variance_model(self, cdataset):
        # Check that the correct variance model is selected for dataset
        session = bmds.BMDS.version("BMDS270", bmds.constants.CONTINUOUS, dataset=cdataset)
        for model in session.model_options:
            session.add_model(bmds.constants.M_Power)
        session.execute()
        model = session.models[0]
        anova = cdataset.anova()
        calc_pvalue2 = anova.test2.TEST
        correct_pvalue2 = model.output["p_value2"]
        # large tolerance due to reporting in text-file
        atol = 1e-3
        assert np.isclose(calc_pvalue2, correct_pvalue2, atol=atol)

    def test_anova(self, anova_dataset, bad_anova_dataset):
        # Check that anova generates expected output from original specifications.
        report = anova_dataset.get_anova_report()
        expected = "                     Tests of Interest    \n   Test    -2*log(Likelihood Ratio)  Test df        p-value    \n   Test 1              22.2699         12           0.0346\n   Test 2               5.5741          6           0.4725\n   Test 3               5.5741          6           0.4725"  # noqa
        assert report == expected

        # check bad anova dataset
        report = bad_anova_dataset.get_anova_report()
        expected = "ANOVA cannot be calculated for this dataset."
        assert report == expected

    def test_dfile_outputs(self):
        ds = bmds.ContinuousDataset(doses=dummy3, ns=dummy3, means=dummy3, stdevs=dummy3)
        dfile = ds.as_dfile()
        expected = "Dose NumAnimals Response Stdev\n1.000000 1 1.000000 1.000000\n2.000000 2 2.000000 2.000000\n3.000000 3 3.000000 3.000000"  # noqa
        assert dfile == expected

    def test_dataset_reporting_options(self, cdataset):
        # test defaults
        assert cdataset._get_dose_units_text() == ""
        assert cdataset._get_response_units_text() == ""
        assert cdataset._get_dataset_name() == "BMDS output results"

        # test settings
        ds = bmds.ContinuousDataset(
            doses=[0, 10, 50, 150, 400],
            ns=[111, 142, 143, 93, 42],
            means=[2.112, 2.095, 1.956, 1.587, 1.254],
            stdevs=[0.235, 0.209, 0.231, 0.263, 0.159],
            dose_units="μg/m³",
            response_units="mg/kg",
            name="Smith 2017: Relative Liver Weight in Male SD Rats",
        )
        assert ds._get_dataset_name() == "Smith 2017: Relative Liver Weight in Male SD Rats"
        assert ds._get_dose_units_text() == " (μg/m³)"
        assert ds._get_response_units_text() == " (mg/kg)"

    def test_dose_drops(self, cidataset):
        cdataset = bmds.ContinuousDataset(
            doses=list(reversed([0, 10, 50, 150, 400])),
            ns=list(reversed([111, 142, 143, 93, 42])),
            means=list(reversed([2.112, 2.095, 1.956, 1.587, 1.254])),
            stdevs=list(reversed([0.235, 0.209, 0.231, 0.263, 0.159])),
        )

        assert (
            cdataset.as_dfile()
            == "Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000\n400.000000 42 1.254000 0.159000"
        )  # noqa
        cdataset.drop_dose()
        assert (
            cdataset.as_dfile()
            == "Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000\n150.000000 93 1.587000 0.263000"
        )  # noqa
        cdataset.drop_dose()
        assert (
            cdataset.as_dfile()
            == "Dose NumAnimals Response Stdev\n0.000000 111 2.112000 0.235000\n10.000000 142 2.095000 0.209000\n50.000000 143 1.956000 0.231000"
        )  # noqa
        with pytest.raises(ValueError):
            cdataset.drop_dose()

    def test_serialize(self):
        ds1 = bmds.ContinuousDataset(
            doses=[1, 2, 3, 4],
            ns=[1, 2, 3, 4],
            means=[1, 2, 3, 4],
            stdevs=[1, 2, 3, 4],
            id=123,
            name="test",
        )

        # make sure serialize looks correct
        serialized = ds1.serialize()
        assert serialized.dict(exclude={"plotting"}) == {
            "dtype": "C",
            "metadata": {
                "id": 123,
                "name": "test",
                "dose_units": "",
                "response_units": "",
                "dose_name": "",
                "response_name": "",
            },
            "doses": [1.0, 2.0, 3.0, 4.0],
            "ns": [1, 2, 3, 4],
            "means": [1.0, 2.0, 3.0, 4.0],
            "stdevs": [1.0, 2.0, 3.0, 4.0],
            "anova": None,
        }

        # make sure we get the correct class back
        ds2 = serialized.deserialize()
        assert isinstance(ds2, bmds.ContinuousDataset)
        assert not isinstance(ds2, bmds.ContinuousIndividualDataset)

        # make sure we get the same result back after deserializing
        assert ds1.serialize().dict() == ds2.serialize().dict()


class TestContinuousIndividualDataset:
    def test_validation(self):
        # these should be valid
        bmds.ContinuousIndividualDataset(doses=dummy3, responses=dummy3)
        # these should raise errors
        with pytest.raises((IndexError, ValueError)):
            # different sized lists
            bmds.ContinuousIndividualDataset(doses=dummy4, responses=dummy3)
            # also duplicate, but less than 2 dose-groups
            bmds.ContinuousIndividualDataset(doses=dummy3_dups, responses=dummy3)

    def test_extra_kwargs(self):
        ds = bmds.ContinuousIndividualDataset(
            doses=[0, 0, 1, 1, 2, 2, 3, 3],
            responses=[8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567],
        )
        assert ds.to_dict()["metadata"] == {
            "id": None,
            "name": "",
            "dose_units": "",
            "response_units": "",
            "dose_name": "",
            "response_name": "",
        }

        assert ds.get_xlabel() == "Dose"
        assert ds.get_ylabel() == "Response"

        ds = bmds.ContinuousIndividualDataset(
            doses=[0, 0, 1, 1, 2, 2, 3, 3],
            responses=[8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567],
            id=123,
            name="example dataset",
            dose_units="mg/kg/d",
            response_units="ug/m3",
            dose_name="Intravenous",
            response_name="Volume",
        )
        assert ds.to_dict()["metadata"] == {
            "id": 123,
            "name": "example dataset",
            "dose_units": "mg/kg/d",
            "response_units": "ug/m3",
            "dose_name": "Intravenous",
            "response_name": "Volume",
        }

        assert ds.get_xlabel() == "Intravenous (mg/kg/d)"
        assert ds.get_ylabel() == "Volume (ug/m3)"

    def test_dfile_outputs(self):
        ds = bmds.ContinuousIndividualDataset(doses=dummy3, responses=dummy3)
        dfile = ds.as_dfile()
        expected = "Dose Response\n1.000000 1.000000\n2.000000 2.000000\n3.000000 3.000000"
        assert dfile == expected

    def test_ci_summary_stats(self, cidataset):
        assert len(cidataset.doses) == 7
        assert np.isclose(cidataset.ns, [8, 6, 6, 6, 6, 6, 6]).all()
        assert np.isclose(
            cidataset.means, [9.9264, 10.1889, 10.17755, 10.3571, 10.0275, 11.4933, 10.85275]
        ).all()
        assert np.isclose(
            cidataset.stdevs, [0.87969, 0.90166, 0.50089, 0.85590, 0.42833, 0.83734, 0.690373]
        ).all()

    def test_dose_drops(self, cidataset):
        assert (
            cidataset.as_dfile()
            == "Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500\n300.000000 10.368000\n300.000000 10.517600\n300.000000 11.316800\n300.000000 12.002000\n300.000000 12.118600\n300.000000 12.636800\n500.000000 9.957200\n500.000000 10.134700\n500.000000 10.774300\n500.000000 11.057100\n500.000000 11.156400\n500.000000 12.036800"
        )  # noqa
        cidataset.drop_dose()
        assert (
            cidataset.as_dfile()
            == "Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500\n300.000000 10.368000\n300.000000 10.517600\n300.000000 11.316800\n300.000000 12.002000\n300.000000 12.118600\n300.000000 12.636800"
        )  # noqa
        cidataset.drop_dose()
        assert (
            cidataset.as_dfile()
            == "Dose Response\n0.000000 8.107900\n0.000000 9.306300\n0.000000 9.743100\n0.000000 9.781400\n0.000000 10.051700\n0.000000 10.613200\n0.000000 10.750900\n0.000000 11.056700\n0.100000 9.155600\n0.100000 9.682100\n0.100000 9.825600\n0.100000 10.209500\n0.100000 10.222200\n0.100000 12.038200\n1.000000 9.566100\n1.000000 9.705900\n1.000000 9.990500\n1.000000 10.271600\n1.000000 10.471000\n1.000000 11.060200\n10.000000 8.851400\n10.000000 10.010700\n10.000000 10.085400\n10.000000 10.568300\n10.000000 11.139400\n10.000000 11.487500\n100.000000 9.542700\n100.000000 9.721100\n100.000000 9.826700\n100.000000 10.023100\n100.000000 10.183300\n100.000000 10.868500"
        )  # noqa
        cidataset.drop_dose()
        cidataset.drop_dose()
        with pytest.raises(ValueError):
            cidataset.drop_dose()

    def test_serialize(self):
        ds1 = bmds.ContinuousIndividualDataset(
            doses=[0, 0, 1, 1, 2, 2, 3, 3],
            responses=[8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567],
            id=123,
            name="test",
        )

        # make sure serialize looks correct
        serialized = ds1.serialize()
        assert serialized.dict(exclude={"anova"}) == {
            "dtype": "CI",
            "metadata": {
                "id": 123,
                "name": "test",
                "dose_units": "",
                "response_units": "",
                "dose_name": "",
                "response_name": "",
            },
            "doses": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "responses": [8.1079, 9.3063, 9.7431, 9.7814, 10.0517, 10.6132, 10.7509, 11.0567],
        }

        # make sure we get the correct class back
        ds2 = serialized.deserialize()
        assert not isinstance(ds2, bmds.ContinuousDataset)
        assert isinstance(ds2, bmds.ContinuousIndividualDataset)

        # make sure we get the same result back after deserializing
        assert ds1.serialize().dict() == ds2.serialize().dict()


class TestContinuousDatasetSchema:
    def test_schema(self, cdataset):
        # check that cycling through serialization returns the same
        v1 = cdataset.serialize().dict()
        v2 = bmds.ContinuousDatasetSchema.parse_obj(v1).deserialize().serialize().dict()
        assert v1 == v2

        data = deepcopy(v1)
        data["doses"] = data["doses"][:-1]
        with pytest.raises(
            ValidationError,
            match="Length of doses, ns, means, and stdevs are not the same",
        ):
            bmds.ContinuousDatasetSchema.parse_obj(data)

        data = deepcopy(v1)
        data.update(ns=[1, 2])
        with pytest.raises(ValidationError, match="Length"):
            bmds.ContinuousDatasetSchema.parse_obj(data)

        data = deepcopy(v1)
        data.update(doses=[0, 10], ns=[20, 20], means=[0, 0], stdevs=[1, 1])
        with pytest.raises(ValidationError, match="At least 3 groups are required"):
            bmds.ContinuousDatasetSchema.parse_obj(data)


class TestContinuousIndividualDatasetSchema:
    def test_schema(self, cidataset):
        # check that cycling through serialization returns the same
        v1 = cidataset.serialize().dict()
        v2 = bmds.ContinuousIndividualDatasetSchema.parse_obj(v1).deserialize().serialize().dict()
        assert v1 == v2

        data = deepcopy(v1)
        data["doses"] = data["doses"][:-1]
        with pytest.raises(
            ValidationError,
            match="Length of doses and responses are not the same",
        ):
            bmds.ContinuousIndividualDatasetSchema.parse_obj(data)

        data = deepcopy(v1)
        data["doses"] = [1, 2]
        with pytest.raises(ValidationError, match="Length of doses and responses are not the same"):
            bmds.ContinuousIndividualDatasetSchema.parse_obj(data)

        data = deepcopy(v1)
        data.update(doses=[0, 10], responses=[20, 20])
        with pytest.raises(ValidationError, match="At least 3 groups are required"):
            bmds.ContinuousIndividualDatasetSchema.parse_obj(data)
