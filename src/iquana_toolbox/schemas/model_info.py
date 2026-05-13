from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from iquana_toolbox.schemas.database.labels import LabelHierarchy, Label


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
    label: Label = Field(..., description="The label that the model can predict.")


class SemanticSegmentationModelInfo(ModelInfo):
    """ Extends ModelInfo to provide semantic segmentation specific information. This is deprecated! """
