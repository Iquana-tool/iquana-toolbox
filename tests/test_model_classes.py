"""A model's label space (ModelInfo.classes) and the class a prediction names (Contour.model_class)."""
import json

import pytest
from pydantic import ValidationError

from iquana_toolbox.mlflow import MLFlowModelRegistry
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import (
    InstanceSegmentationModelInfo,
    ModelClass,
    ModelInfo,
    parse_tags_to_model_info,
)

CLASSES = [
    ModelClass(key="region", name="Choroid region"),
    ModelClass(key="vessel", name="Choroidal vessels", part_of="region"),
]


def _info(**kwargs):
    return InstanceSegmentationModelInfo(
        registry_key="m", name="M", description="d", usage_tip=None, **kwargs
    )


def test_classes_default_to_class_agnostic():
    assert _info().classes == []


def test_classes_survive_the_tag_round_trip():
    tags = MLFlowModelRegistry._model_info_tags(_info(classes=CLASSES))
    assert json.loads(tags["classes"])[1]["part_of"] == "region"
    parsed = parse_tags_to_model_info({
        **tags, "registry_key": "m", "name": "M", "description": "d", "usage_tip": None,
        "task": "instance-segmentation",
    })
    assert parsed.classes == CLASSES


def test_class_keys_must_be_unique():
    with pytest.raises(ValidationError, match="unique keys"):
        _info(classes=[ModelClass(key="a", name="A"), ModelClass(key="a", name="B")])


def test_part_of_must_name_a_declared_class():
    with pytest.raises(ValidationError, match="unknown part_of"):
        ModelInfo(registry_key="m", name="M", description="d", usage_tip=None,
                  classes=[ModelClass(key="vessel", name="V", part_of="region")])


def test_contour_carries_its_model_class():
    contour = Contour(x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2], model_class="vessel")
    assert Contour.model_validate(contour.model_dump()).model_class == "vessel"
    assert Contour(x=[], y=[]).model_class is None
