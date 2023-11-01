import ctypes
from itertools import cycle

from pydantic import Field

from ... import plotting
from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import DichotomousModelAverageResult
from ..types.structs import DichotomousMAStructs
from .base import BmdModelAveraging, BmdModelAveragingSchema, InputModelSettings


class BmdModelAveragingDichotomous(BmdModelAveraging):
    model_version: str = "BMDS330"

    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        if settings is None:
            return DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            return settings
        else:
            return DichotomousModelSettings.model_validate(settings)

    def execute(self) -> DichotomousModelAverageResult:
        structs = DichotomousMAStructs.from_session(
            self.session.dataset, self.models, self.session.ma_weights
        )
        self.structs = structs

        dll = self.get_dll()
        dll.runBMDSDichoMA(
            ctypes.pointer(structs.analysis),
            ctypes.pointer(structs.inputs),
            ctypes.pointer(structs.dich_result),
            ctypes.pointer(structs.result),
        )
        self.results = DichotomousModelAverageResult.from_structs(
            structs, [model.results for model in self.models]
        )
        return self.results

    def serialize(self, session) -> "BmdModelAveragingDichotomousSchema":
        model_indexes = [session.models.index(model) for model in self.models]
        return BmdModelAveragingDichotomousSchema(
            settings=self.settings, model_indexes=model_indexes, results=self.results
        )

    def plot(self, colorize: bool = False):
        """
        After model execution, print the dataset, curve-fit, BMD, and BMDL.
        """
        if not self.has_results:
            raise ValueError("Cannot plot if results are unavailable")
        dataset = self.session.dataset
        results = self.results
        fig = dataset.plot()
        ax = fig.gca()
        ax.set_ylim(-0.05, 1.05)
        title = f"{dataset._get_dataset_name()}\nModel average, {self.settings.bmr_text}"
        ax.set_title(title)
        if colorize:
            color_cycle = cycle(plotting.INDIVIDUAL_MODEL_COLORS)
            line_cycle = cycle(plotting.INDIVIDUAL_LINE_STYLES)
        else:
            color_cycle = cycle(["#ababab"])
            line_cycle = cycle(["solid"])
        for i, model in enumerate(self.session.models):
            if colorize:
                label = model.name()
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
        ax.plot(
            self.results.dr_x,
            self.results.dr_y,
            label="Model average (BMD, BMDL, BMDU)",
            c="#6470C0",
            lw=4,
            zorder=110,
        )
        plotting.add_bmr_lines(ax, results.bmd, results.bmd_y, results.bmdl, results.bmdu)

        # reorder handles and labels
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 1]
        ax.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order], **plotting.LEGEND_OPTS
        )

        return fig


class BmdModelAveragingDichotomousSchema(BmdModelAveragingSchema):
    settings: DichotomousModelSettings
    results: DichotomousModelAverageResult
    bmds_model_indexes: list[int] = Field(alias="model_indexes")

    def deserialize(self, session) -> BmdModelAveragingDichotomous:
        models = [session.models[idx] for idx in self.bmds_model_indexes]
        ma = BmdModelAveragingDichotomous(session=session, models=models, settings=self.settings)
        ma.results = self.results
        return ma
