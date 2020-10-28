class DatasetBase:
    # Abstract parent-class for dataset-types.

    def _validate(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def as_dfile(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def to_dict(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def plot(self):
        raise NotImplementedError("Abstract method; requires implementation")

    def drop_dose(self):
        raise NotImplementedError("Abstract method; requires implementation")

    @property
    def num_dose_groups(self):
        return len(set(self.doses))

    def _get_dose_units_text(self):
        return " ({})".format(self.kwargs["dose_units"]) if "dose_units" in self.kwargs else ""

    def _get_response_units_text(self):
        return (
            " ({})".format(self.kwargs["response_units"]) if "response_units" in self.kwargs else ""
        )

    def _get_dataset_name(self):
        return self.kwargs.get("dataset_name", "BMDS output results")
