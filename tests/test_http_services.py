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


def test_propagation_request_frame_range_is_clamped_and_ordered():
    import numpy as np
    from iquana_toolbox.schemas.database.masks import BinaryMask
    from iquana_toolbox.schemas.networking.http.services import (
        FrameMask, PropagationObject, PropagationRequest,
    )

    mask = BinaryMask.from_numpy_array(np.ones((4, 4), dtype=np.uint8))
    request = PropagationRequest(
        image_url="frame_2.png", user_id="u", model_registry_key="sam2-1-tiny",
        frame_urls=[f"frame_{i}.png" for i in range(5)],
        objects=[PropagationObject(object_id=7, frame_masks=[FrameMask(frame_index=2, mask=mask)])],
        start_frame=9, end_frame=1,
    )
    assert request.frame_range == (1, 4)
    assert request.direction == "both"
    assert PropagationRequest.model_validate_json(request.model_dump_json()).objects[0].object_id == 7
