import json

import pytest
from pydantic import ValidationError

from iquana_toolbox.schemas import ConditioningSpec, InputContract, get_contract_for_task
from iquana_toolbox.schemas.model_info import ModelInfo, parse_tags_to_model_info
from iquana_toolbox.schemas.training import HyperParameter


def _parameter(**overrides) -> HyperParameter:
    values = {
        "key": "threshold",
        "label": "Threshold",
        "type": "float",
        "default_value": 0.3,
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.05,
    }
    values.update(overrides)
    return HyperParameter(**values)


def _contract(task: str = "instance-suggestion", **overrides) -> InputContract:
    values = {
        "task": task,
        "conditioning": ConditioningSpec(
            kind="instances",
            unit="instance",
            min_units=1,
            user_selectable_count=False,
        ),
        "parameters": [_parameter()],
    }
    values.update(overrides)
    return InputContract(**values)


def test_valid_contract_round_trips_through_json() -> None:
    contract = _contract(notes="Detection controls")

    payload = json.loads(json.dumps(contract.model_dump(mode="json")))

    assert InputContract.model_validate(payload) == contract
    assert payload["schema_version"] == 1


def test_conditioning_kind_requires_its_declared_unit() -> None:
    with pytest.raises(ValidationError, match="requires unit='image'"):
        ConditioningSpec(kind="reference_images", unit=None)

    with pytest.raises(ValidationError, match="must use unit='vector'"):
        ConditioningSpec(kind="embeddings", unit="image", embedding_kinds=["region_mean"])


def test_non_counted_conditioning_has_no_unit_or_count_control() -> None:
    assert ConditioningSpec(kind="none", user_selectable_count=False).unit is None
    assert ConditioningSpec(kind="concept_text", user_selectable_count=False).min_units == 0

    with pytest.raises(ValidationError, match="must not declare a unit"):
        ConditioningSpec(kind="none", unit="instance", user_selectable_count=False)
    with pytest.raises(ValidationError, match="must set user_selectable_count=False"):
        ConditioningSpec(kind="concept_text")


def test_conditioning_cardinality_and_annotation_rules_are_validated() -> None:
    with pytest.raises(ValidationError, match="min_units.*must not exceed"):
        ConditioningSpec(kind="instances", unit="instance", min_units=2, max_units=1)

    with pytest.raises(ValidationError, match="only valid for.*reference_images"):
        ConditioningSpec(kind="instances", unit="instance", requires_complete_annotation=True)

    assert ConditioningSpec(
        kind="reference_images",
        unit="image",
        requires_complete_annotation=True,
    ).requires_complete_annotation


def test_embeddings_require_non_empty_unique_embedding_kinds() -> None:
    assert ConditioningSpec(
        kind="embeddings",
        unit="vector",
        embedding_kinds=["region_mean"],
    )

    with pytest.raises(ValidationError, match="requires at least one"):
        ConditioningSpec(kind="embeddings", unit="vector")
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ConditioningSpec(
            kind="embeddings",
            unit="vector",
            embedding_kinds=["region_mean", "region_mean"],
        )
    with pytest.raises(ValidationError, match="only valid for.*embeddings"):
        ConditioningSpec(kind="instances", unit="instance", embedding_kinds=["region_mean"])


def test_parameter_keys_must_be_unique_and_non_blank() -> None:
    with pytest.raises(ValidationError, match="duplicate inference parameter key"):
        _contract(parameters=[_parameter(), _parameter()])

    with pytest.raises(ValidationError, match="keys must not be blank"):
        _contract(parameters=[_parameter(key=" ")])


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (_parameter(type="int", default_value=1.2), "does not match type 'int'"),
        (_parameter(default_value=2.0, max_value=1.0), "above max_value"),
        (_parameter(options=[0.1, 0.2]), "must be one of options"),
        (_parameter(options=[]), "options must not be empty"),
        (_parameter(min_value=0.8, max_value=0.2), "min_value must not exceed"),
        (_parameter(step=0), "step must be positive"),
        (_parameter(type="str", default_value="auto", min_value=0), "numeric constraints"),
    ],
)
def test_parameter_semantics_are_validated(parameter: HyperParameter, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _contract(parameters=[parameter])


def test_model_info_rejects_duplicate_contract_tasks() -> None:
    with pytest.raises(ValidationError, match="at most one contract per task"):
        ModelInfo(
            registry_key="demo",
            name="Demo",
            description="Demo model",
            usage_tip="Use it",
            input_contracts=[_contract(), _contract()],
        )


def test_model_info_and_parser_preserve_contracts() -> None:
    contract = _contract()
    info = ModelInfo(
        registry_key="demo",
        name="Demo",
        description="Demo model",
        usage_tip="Use it",
        tags={"task": "instance-suggestion"},
        input_contracts=[contract],
    )

    parsed = parse_tags_to_model_info(
        {
            "registry_key": info.registry_key,
            "name": info.name,
            "description": info.description,
            "usage_tip": info.usage_tip,
            "task": "instance-suggestion",
            "input_contracts": json.dumps(info.model_dump(mode="json")["input_contracts"]),
        }
    )

    assert parsed.input_contracts == [contract]
    assert get_contract_for_task(parsed.input_contracts, "instance-suggestion") == contract


def test_legacy_model_info_without_contracts_still_parses() -> None:
    parsed = parse_tags_to_model_info(
        {
            "registry_key": "legacy",
            "name": "Legacy",
            "description": "Legacy model",
            "usage_tip": "Use it",
        }
    )

    assert parsed.input_contracts == []


@pytest.mark.parametrize("malformed", ["", None, "{\"task\": \"demo\"}"])
def test_malformed_declared_contracts_do_not_become_legacy_defaults(malformed) -> None:
    with pytest.raises(ValidationError):
        parse_tags_to_model_info(
            {
                "registry_key": "broken",
                "name": "Broken",
                "description": "Broken model",
                "usage_tip": "Do not use",
                "input_contracts": malformed,
            }
        )
