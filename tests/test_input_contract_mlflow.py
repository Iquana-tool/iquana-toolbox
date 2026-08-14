import json

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
