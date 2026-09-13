"""Model-declared inference input contracts.

An :class:`InputContract` is the model-owned description of the inputs for
one advertised inference task.  It deliberately reuses the toolbox's
``HyperParameter`` descriptor so the frontend can render training and
inference controls from the same shape.

The schema is intentionally strict about semantic combinations.  Pydantic's
field validation can tell us that ``max_units`` is an integer, but only the
cross-field validators below can tell us that a ``none`` contract must not
claim to consume instances or that an embedding contract must name the
embedding kinds it needs.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from iquana_toolbox.schemas.training import HyperParameter


ConditioningKind = Literal[
    "instances",
    "reference_images",
    "embeddings",
    "concept_text",
    "none",
]
ConditioningUnit = Literal["instance", "image", "vector"]


class ConditioningSpec(BaseModel):
    """Describe the conditioning data consumed by a model task.

    ``unit`` is the unit counted by ``min_units`` and ``max_units``.  It is
    absent for conditioning kinds whose cardinality is not meaningful
    (``concept_text`` and ``none``).
    """

    kind: ConditioningKind = Field(
        ...,
        description="The kind of conditioning data consumed by the model.",
    )
    unit: Optional[ConditioningUnit] = Field(
        default=None,
        description="The unit counted by min_units and max_units.",
    )
    min_units: int = Field(
        default=0,
        ge=0,
        description="Minimum number of conditioning units required.",
    )
    max_units: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of conditioning units; null means unbounded.",
    )
    requires_complete_annotation: bool = Field(
        default=False,
        description=(
            "Whether reference images must be fully annotated for the requested "
            "concept."
        ),
    )
    embedding_kinds: list[str] = Field(
        default_factory=list,
        description="Embedding kinds required when kind is embeddings.",
    )
    user_selectable_count: bool = Field(
        default=True,
        description="Whether the UI should expose the conditioning count.",
    )

    @model_validator(mode="after")
    def validate_semantics(self) -> "ConditioningSpec":
        counted_units: dict[str, ConditioningUnit] = {
            "instances": "instance",
            "reference_images": "image",
            "embeddings": "vector",
        }

        if self.kind in counted_units:
            expected_unit = counted_units[self.kind]
            if self.unit is None:
                raise ValueError(
                    f"conditioning kind '{self.kind}' requires unit='{expected_unit}'"
                )
            if self.unit != expected_unit:
                raise ValueError(
                    f"conditioning kind '{self.kind}' must use unit='{expected_unit}', "
                    f"not '{self.unit}'"
                )
        else:
            if self.unit is not None:
                raise ValueError(
                    f"conditioning kind '{self.kind}' must not declare a unit"
                )
            if self.min_units != 0 or self.max_units is not None:
                raise ValueError(
                    f"conditioning kind '{self.kind}' does not have a unit cardinality; "
                    "use min_units=0 and max_units=null"
                )
            if self.user_selectable_count:
                raise ValueError(
                    f"conditioning kind '{self.kind}' must set "
                    "user_selectable_count=False"
                )

        if self.max_units is not None and self.min_units > self.max_units:
            raise ValueError(
                f"min_units ({self.min_units}) must not exceed "
                f"max_units ({self.max_units})"
            )

        if self.requires_complete_annotation and self.kind != "reference_images":
            raise ValueError(
                "requires_complete_annotation is only valid for "
                "conditioning kind 'reference_images'"
            )

        if self.kind == "embeddings":
            if not self.embedding_kinds:
                raise ValueError(
                    "conditioning kind 'embeddings' requires at least one "
                    "embedding_kinds entry"
                )
            if any(not isinstance(kind, str) or not kind.strip() for kind in self.embedding_kinds):
                raise ValueError("embedding_kinds entries must be non-empty strings")
            if len(set(self.embedding_kinds)) != len(self.embedding_kinds):
                raise ValueError("embedding_kinds must not contain duplicates")
        elif self.embedding_kinds:
            raise ValueError(
                "embedding_kinds is only valid for conditioning kind 'embeddings'"
            )

        return self


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _value_matches_type(value: Any, parameter_type: str) -> bool:
    """Check a parameter value without relying on Pydantic coercion."""
    if parameter_type == "bool":
        return isinstance(value, bool)
    if parameter_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter_type == "float":
        return _is_finite_number(value)
    if parameter_type == "str":
        return isinstance(value, str)
    return False


class InputContract(BaseModel):
    """Declare the complete inference input surface for one model task."""

    schema_version: Literal[1] = Field(
        default=1,
        description="Version of the contract schema; currently only version 1 exists.",
    )
    task: str = Field(
        ...,
        min_length=1,
        description="The advertised task this contract applies to.",
    )
    conditioning: ConditioningSpec = Field(
        ...,
        description="The conditioning data consumed by this task.",
    )
    parameters: list[HyperParameter] = Field(
        default_factory=list,
        description="Inference parameters accepted by this task.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Human-readable guidance for the UI or API consumer.",
    )

    @model_validator(mode="after")
    def validate_semantics(self) -> "InputContract":
        if not self.task.strip():
            raise ValueError("task must not be blank")

        seen_keys: set[str] = set()
        for parameter in self.parameters:
            key = parameter.key
            if not key.strip():
                raise ValueError("inference parameter keys must not be blank")
            if key in seen_keys:
                raise ValueError(f"duplicate inference parameter key: '{key}'")
            seen_keys.add(key)
            self._validate_parameter(parameter)

        return self

    @staticmethod
    def _validate_parameter(parameter: HyperParameter) -> None:
        parameter_type = parameter.type
        if not _value_matches_type(parameter.default_value, parameter_type):
            raise ValueError(
                f"parameter '{parameter.key}' default_value does not match type "
                f"'{parameter_type}'"
            )

        if parameter.options is not None:
            if not parameter.options:
                raise ValueError(f"parameter '{parameter.key}' options must not be empty")
            invalid_options = [
                option
                for option in parameter.options
                if not _value_matches_type(option, parameter_type)
            ]
            if invalid_options:
                raise ValueError(
                    f"parameter '{parameter.key}' options must match type "
                    f"'{parameter_type}'"
                )
            if parameter.default_value not in parameter.options:
                raise ValueError(
                    f"parameter '{parameter.key}' default_value must be one of options"
                )

        has_numeric_constraints = any(
            value is not None
            for value in (parameter.min_value, parameter.max_value, parameter.step)
        )
        if parameter_type not in {"int", "float"}:
            if has_numeric_constraints:
                raise ValueError(
                    f"parameter '{parameter.key}' uses numeric constraints but has "
                    f"type '{parameter_type}'"
                )
            return

        for field_name, value in (
            ("min_value", parameter.min_value),
            ("max_value", parameter.max_value),
            ("step", parameter.step),
        ):
            if value is not None and not _is_finite_number(value):
                raise ValueError(
                    f"parameter '{parameter.key}' {field_name} must be finite"
                )

        if (
            parameter.min_value is not None
            and parameter.max_value is not None
            and parameter.min_value > parameter.max_value
        ):
            raise ValueError(
                f"parameter '{parameter.key}' min_value must not exceed max_value"
            )
        if parameter.step is not None and parameter.step <= 0:
            raise ValueError(f"parameter '{parameter.key}' step must be positive")

        default_value = parameter.default_value
        if not _is_finite_number(default_value):
            raise ValueError(f"parameter '{parameter.key}' default_value must be finite")
        if parameter.min_value is not None and default_value < parameter.min_value:
            raise ValueError(
                f"parameter '{parameter.key}' default_value is below min_value"
            )
        if parameter.max_value is not None and default_value > parameter.max_value:
            raise ValueError(
                f"parameter '{parameter.key}' default_value is above max_value"
            )


def get_contract_for_task(
    contracts: Sequence[InputContract], task: str
) -> Optional[InputContract]:
    """Return the contract for *task*, or ``None`` when it is undeclared."""
    for contract in contracts:
        if contract.task == task:
            return contract
    return None
