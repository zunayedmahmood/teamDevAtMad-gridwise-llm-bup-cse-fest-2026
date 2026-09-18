from pydantic import BaseModel, ConfigDict

from app.models.response import DirectiveType


class StrictLLMModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class LLMDirective(StrictLLMModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    hours: list[int] | None
    factor: float | None
    minimum_energy_kwh: float | None
    max_grid_kwh: float | None
    explanation: str


class LLMDirectiveBatch(StrictLLMModel):
    directives: list[LLMDirective]
