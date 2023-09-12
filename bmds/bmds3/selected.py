from typing import Annotated

from pydantic import BaseModel, Field

from .models.base import BmdModel


class SelectedModelSchema(BaseModel):
    bmds_model_index: Annotated[int, Field(ge=0)] | None = None
    notes: str = ""

    def deserialize(self, session) -> "SelectedModel":
        return SelectedModel(session, bmds_model_index=self.bmds_model_index, notes=self.notes)


class SelectedModel:
    def __init__(self, session, bmds_model_index: int | None = None, notes: str = ""):
        self.session = session
        self.bmds_model_index = bmds_model_index
        self.notes = notes

    def select(self, model: BmdModel | None, notes: str):
        self.bmds_model_index = self.session.models.index(model) if model is not None else None
        self.notes = notes

    @property
    def model(self) -> BmdModel | None:
        """Returns the selected model if one exists, else None"""
        if self.bmds_model_index is not None:
            return self.session.models[self.bmds_model_index]
        return None

    @property
    def no_model_selected(self) -> bool:
        """Check if no model selected was deliberate or just undefined.

        Assumes that if a user has provided notes then that is a clear decision that not model was
        selected; otherwise it's likely that a selection is just undefined.
        """
        return self.bmds_model_index is None and isinstance(self.notes, str)

    def serialize(self) -> SelectedModelSchema:
        return SelectedModelSchema(bmds_model_index=self.bmds_model_index, notes=self.notes)

    def update_record(self, d: dict, index: int) -> None:
        """Update data record for a tabular-friendly export"""
        is_selected = self.bmds_model_index == index
        d.update(
            selected=is_selected,
            selected_notes=self.notes if is_selected else None,
        )
