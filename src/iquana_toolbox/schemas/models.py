from typing import List, Optional

from pydantic import BaseModel, Field

from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.training import TrainingProgress


class BaseAIModel(BaseModel):
    registry_key: str = Field(
        ...,
        examples=["unet", "sam2.1_tiny"],
        description="A key used to retrieve the model from a registry. Every AI model "
                    "must have one.")
    name: str = Field(..., description="Human-readable name of the model. ")
    description: str = Field(..., description="Human-readable description of the model. "
                                              "Gives more information about the model.")
    tags: List[str] = Field(...,
                            description="Human-readable tags of the model. Tags are short descriptors of the model.")
    number_of_parameters: int | None = Field(...,
                                             description="The number of parameters in the model.")
    pretrained: bool = Field(..., description="Whether or not the model is trained on pretrained models.")
    trainable: bool = Field(..., description="Whether or not the model is trainable.")
    finetunable: bool = Field(..., description="Whether or not the model is finetunable.")


class PromptedSegmentationModels(BaseAIModel):
    prompt_types_supported: list = Field(..., description="A list of prompt types supported by the model.")
    refinement_supported: bool = Field(..., description="Whether or not the model supports refinement.")


class CompletionModel(BaseAIModel):
    pass


class SemanticSegmentationModels(BaseAIModel):
    label_hierarchy: Optional[LabelHierarchy] = Field(
        default=None,
        description="Label hierarchy that the model can predict. This does not mean"
                    "the model predicts hierarchical segments. It is used to check "
                    "what labels can be predicted in general. Eg. when the user makes"
                    "significant changes to the label hierarchy, the model becomes "
                    "deprecated, as it is not trained to predict this. It is optional,"
                    "because base models dont predict anything."
    )
    training_task_id: Optional[int] | None = Field(default=None, description="The id of the celery task for training.")
    progress: Optional[TrainingProgress] | None = Field(
        default=None,
        description="A class to track the progress of training and get the history of values."
    )

    def is_base_model(self):
        return self.label_hierarchy is None
