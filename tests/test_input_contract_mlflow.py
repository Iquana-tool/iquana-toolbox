import json
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo, parse_tags_to_model_info


def test_mlflow_discovery_tags_serialize_contracts_without_losing_types() -> None:
    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images",
            unit="image",
            min_units=1,
            max_units=1,
            requires_complete_annotation=True,
            user_selectable_count=False,
        ),
    )
    info = ModelInfo(
        registry_key="demo",
        name="Demo",
        description="Demo model",
        usage_tip="Use one reference image",
        tags={"task": "cross-image-suggestion"},
        input_contracts=[contract],
    )

    tags = MLFlowModelRegistry._model_info_tags(info)
    decoded = json.loads(tags["input_contracts"])
    parsed = parse_tags_to_model_info(
        {
            "registry_key": info.registry_key,
            "name": info.name,
            "description": info.description,
            "usage_tip": info.usage_tip,
            **tags,
        }
    )

    assert decoded[0]["conditioning"]["max_units"] == 1
    assert decoded[0]["conditioning"]["requires_complete_annotation"] is True
    assert parsed.input_contracts == [contract]


def test_mlflow_discovery_tags_leave_legacy_contracts_absent() -> None:
    info = ModelInfo(
        registry_key="legacy",
        name="Legacy",
        description="Legacy model",
        usage_tip="Use it",
        tags={"task": "instance-segmentation"},
    )

    tags = MLFlowModelRegistry._model_info_tags(info)

    assert "input_contracts" not in tags


def test_registered_model_sync_publishes_authoritative_empty_contract_tag() -> None:
    registry = MLFlowModelRegistry("http://example")
    registry.client = MagicMock()
    registry.check_registered = MagicMock(return_value=True)
    model_info = ModelInfo(
        registry_key="demo",
        name="Demo",
        description="Demo model",
        usage_tip="Use it",
        tags={"task": "instance-segmentation", "team": "cv"},
        input_contracts=[],
    )

    with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri"):
        registry.register_model(SimpleNamespace(model_info=model_info))

    registry.client.set_registered_model_tag.assert_any_call("demo", "input_contracts", "[]")
    registry.client.delete_registered_model_tag.assert_not_called()
    registry.client.set_registered_model_tag.assert_has_calls(
        [
            call("demo", "task", "instance-segmentation"),
            call("demo", "team", "cv"),
        ],
        any_order=True,
    )


def test_registered_model_sync_writes_empty_contract_tag_when_missing() -> None:
    registry = MLFlowModelRegistry("http://example")
    registry.client = MagicMock()
    registry.check_registered = MagicMock(return_value=True)
    model_info = ModelInfo(
        registry_key="demo",
        name="Demo",
        description="Demo model",
        usage_tip="Use it",
        tags={"task": "instance-segmentation"},
        input_contracts=[],
    )

    with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri"):
        registry.register_model(SimpleNamespace(model_info=model_info))

    registry.client.set_registered_model_tag.assert_any_call("demo", "input_contracts", "[]")
