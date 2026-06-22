import ast
from typing import List, Optional, Literal, Any

from pydantic import BaseModel, Field, ValidationError

from iquana_toolbox.schemas.training import HyperParameter


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _parse_list_like(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return [stripped]
    return [value]


def _parse_dict_like(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): str(val) for key, val in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                return {str(key): str(val) for key, val in parsed.items()}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _canonical_task(task_value: Any) -> Optional[str]:
    if not isinstance(task_value, str):
        return None
    normalized = task_value.strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "prompted_segmentation": "prompted_segmentation",
        "interactive_segmentation": "prompted_segmentation",
        "instance_segmentation": "instance_segmentation",
        "instance_discovery": "instance_discovery",
        "semantic_segmentation": "semantic_segmentation",
    }
    return alias_map.get(normalized)


class ModelInfo(BaseModel):
    """
    A model info object. This is used to gather information about the model, which can be displayed in the frontend.

    Attributes:
        registry_key: str: The registry key of the model. This is used to retrieve the model from a registry.
            Every AI model must have one.
        name: str: Human-readable name of the model. The name to display in the frontend as registry keys can be confusing.
        description: str: Human-readable short description of the model. This describes the model in markdown briefly.

    """
    registry_key: str = Field(
        ...,
        examples=["unet", "sam2.1_tiny"],
        description="A key used to retrieve the model from a registry. Every AI model "
                    "must have one."
    )
    name: str = Field(
        ...,
        description="Human-readable name of the model. "
    )
    description: str = Field(
        ...,
        description="Human-readable description of the model. "
                    "Gives more information about the model in markdown."
    )
    info_url: Optional[str] = Field(
        default=None,
        description="A url to find more information about the model. Could lead to an Arxive paper, a github repository, etc."
    )
    usage_tip: Optional[str] = Field(
        ...,
        description="Human-readable usage tip of the model. E.g. 'Use with bounding boxes', 'Best for medical images' "
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description=
        "Human-readable tags of the model. Tags are short descriptors of the model, e.g."
        "task: instance-segmentation, domain: general, pretrained: true, number_of_parameters: 1M, etc."
    )
    badges: List[str] = Field(
        default_factory=list,
        description=
        "A list of badges to display for the model. Badges are short descriptors of the model, "
        "similar to tags, but meant to be more eye-catching. E.g. 'fast', 'accurate', etc."
    )
    status: Literal["ready", "not_ready"] = Field(
        default="ready",
        description="The status of the model. If 'ready', the model can be used for its respective "
                    "service. Else, the model needs to be trained first."
    )
    trainable: bool = Field(
        default=False,
        description="Whether or not the model is trainable with new data."
    )
    training_parameters: List[HyperParameter] = Field(
        default_factory=list,
        description="The hyperparameters this model exposes for training. The frontend renders "
                    "the training config UI generically from these (defaults, ranges, options)."
    )


class PromptedSegmentationModelInfo(ModelInfo):
    """ Extends ModelInfo to provide prompted segmentation specific information."""
    prompt_types_supported: list = Field(
        ...,
        description="A list of prompt types supported by the model.")
    refinement_supported: bool = Field(
        ...,
        description="Whether or not the model supports refinement. Refinement means that the model can take a previous "
                    "mask as input for another prompting cycle. This is useful for models that support iterative "
                    "prompting, where the user can give multiple prompts to refine the segmentation mask.")


class InstanceDiscoveryModelInfo(ModelInfo):
    """ Extends ModelInfo to provide instance discovery specific information."""
    pass


class InstanceSegmentationModelInfo(ModelInfo):
    """ Extends ModelInfo to provide instance segmentation specific information."""
    label_ids: List[int] = Field(
        default_factory=list,
        description="The label ids (classes) this model predicts. Empty for an untrained base model; "
                    "populated after (multiclass) training with the dataset's labels."
    )


class SemanticSegmentationModelInfo(ModelInfo):
    """ Extends ModelInfo to provide semantic segmentation specific information. This is deprecated! """


def parse_tags_to_model_info(tags: dict[str, Any]) -> ModelInfo:
    if not isinstance(tags, dict):
        raise TypeError("tags must be a dictionary")

    payload: dict[str, Any] = dict(tags)

    if "trainable" in payload:
        parsed_trainable = _parse_optional_bool(payload.get("trainable"))
        if parsed_trainable is not None:
            payload["trainable"] = parsed_trainable

    if "refinement_supported" in payload:
        parsed_refinement = _parse_optional_bool(payload.get("refinement_supported"))
        if parsed_refinement is not None:
            payload["refinement_supported"] = parsed_refinement

    if "label_ids" in payload:
        payload["label_ids"] = [
            parsed for parsed in (_parse_optional_int(v) for v in _parse_list_like(payload.get("label_ids")))
            if parsed is not None
        ]

    if "prompt_types_supported" in payload:
        payload["prompt_types_supported"] = _parse_list_like(payload.get("prompt_types_supported"))

    if "badges" in payload:
        payload["badges"] = [str(badge) for badge in _parse_list_like(payload.get("badges"))]

    if "tags" in payload:
        payload["tags"] = _parse_dict_like(payload.get("tags"))

    task = _canonical_task(payload.get("task"))
    if task is None:
        task = _canonical_task(payload.get("model_task"))

    model_cls: type[ModelInfo] = ModelInfo
    if task == "prompted_segmentation":
        model_cls = PromptedSegmentationModelInfo
    elif task == "instance_segmentation":
        model_cls = InstanceSegmentationModelInfo
    elif task == "instance_discovery":
        model_cls = InstanceDiscoveryModelInfo
    elif task == "semantic_segmentation":
        model_cls = SemanticSegmentationModelInfo
    elif "prompt_types_supported" in payload or "refinement_supported" in payload:
        model_cls = PromptedSegmentationModelInfo
    elif "label_ids" in payload:
        model_cls = InstanceSegmentationModelInfo

    try:
        return model_cls.model_validate(payload)
    except ValidationError:
        if model_cls is ModelInfo:
            raise
        return ModelInfo.model_validate(payload)
