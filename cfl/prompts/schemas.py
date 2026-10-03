from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "2"


class SymbolSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    one_liner: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    inputs: str = Field(min_length=1)
    returns: str = Field(min_length=1)
    side_effects: list[str]
    raises: list[str]
    notable_logic: str = Field(min_length=1)

    @field_validator("one_liner", "purpose", "inputs", "returns", "notable_logic")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Summary fields must not be empty")
        return value.strip()

    @field_validator("one_liner")
    @classmethod
    def trim_one_liner(cls, value: str) -> str:
        return " ".join(value.split()[:25])

    @field_validator("notable_logic")
    @classmethod
    def trim_logic(cls, value: str) -> str:
        return " ".join(value.split()[:60])


SYMBOL_SUMMARY_JSON_SCHEMA = SymbolSummary.model_json_schema()
