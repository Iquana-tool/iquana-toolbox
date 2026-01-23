from typing import List

from pydantic import BaseModel, Field

# Schemas
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
    pass
