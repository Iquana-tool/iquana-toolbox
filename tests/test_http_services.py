import numpy as np

from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.networking.http.services import (
    CrossImageExemplar,
    CrossImageSuggestionRequest,
    InstanceSegmentationRequest,
)


def _exemplar_payload() -> dict:
    mask = BinaryMask.from_numpy_array(np.array([[1]], dtype=np.uint8))
    return {"image_url": "https://example.test/exemplar.png", "mask": mask.model_dump(mode="json")}


def test_source_aware_exemplars_are_shared_by_instance_requests() -> None:
    common = {
        "image_url": "https://example.test/target.png",
        "user_id": 1,
        "model_registry_key": "demo",
    }

    instance_request = InstanceSegmentationRequest(**common, exemplars=[_exemplar_payload()])
    suggestion_request = CrossImageSuggestionRequest(**common, exemplars=[_exemplar_payload()])

    assert isinstance(instance_request.exemplars[0], CrossImageExemplar)
    assert isinstance(suggestion_request.exemplars[0], CrossImageExemplar)


def test_source_aware_exemplars_default_to_independent_empty_lists() -> None:
    common = {
        "image_url": "https://example.test/target.png",
        "user_id": 1,
        "model_registry_key": "demo",
    }

    instance_request = InstanceSegmentationRequest(**common)
    suggestion_request = CrossImageSuggestionRequest(**common)

    assert instance_request.exemplars == []
    assert suggestion_request.exemplars == []
    instance_request.exemplars.append(CrossImageExemplar.model_validate(_exemplar_payload()))
    assert suggestion_request.exemplars == []
