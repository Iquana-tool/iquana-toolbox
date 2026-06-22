from typing import Union, Any, Optional, Literal

from pydantic import BaseModel, Field

from iquana_toolbox.schemas.database.labels import Label


class BaseTrainingRequest(BaseModel):
    dataset_id: int = Field(..., title="Dataset ID", description="The dataset ID.")
    image_folder_path: str = Field(..., description="Path to the folder containing the images.")
    model_registry_key: str = Field(..., description="A key from the model registry")
    user_id: Union[str, int] = Field(..., title="User ID", description="Unique identifier for the user.")
    hyper_parameter: dict = Field(default_factory=dict, description="Hyperparameters of the training.")


class InstanceSegmentationTrainingRequest(BaseTrainingRequest):
    labels: list[Label] = Field(
        ...,
        title="Labels",
        description="The labels (classes) the model should learn to segment. Multiclass by default; "
                    "pass a single label for the single-class edge case.",
    )
    annotation_file_url: str = Field(..., title="Annotation File URL",
                                     description="The path to a COCO annotation file.")


# ---------------------------------------------------------------------------
# Hyperparameter descriptor
#
# Models declare which hyperparameters they expose for training via a list of
# these. One flat shape keeps storage lossless (no pydantic subtype coercion when
# embedded in ``ModelInfo.training_parameters``) and gives the frontend a single
# thing to render. The widget is inferred from which optional fields are set:
#   - ``options`` set            -> dropdown
#   - ``min_value``/``max_value`` set -> slider
#   - otherwise                  -> numeric/text/checkbox input (by ``type``)
# ---------------------------------------------------------------------------

HyperParameterType = Literal["int", "float", "bool", "str"]


class HyperParameter(BaseModel):
    """A single trainable hyperparameter a model exposes to the training UI."""
    key: str = Field(..., title="Key",
                     description="The key passed to the training script (e.g. 'epochs', 'lr').")
    label: str = Field(..., title="Label", description="Human-readable name shown in the UI.")
    default_value: Any = Field(..., title="Default value", description="Default hyperparameter value.")
    description: str = Field("", title="Description", description="Help text shown in the UI.")
    type: HyperParameterType = Field("float", title="Type",
                                     description="Value type, used for input coercion in the UI.")
    options: Optional[list[Any]] = Field(
        default=None, description="If set, the value is chosen from this discrete set (renders a dropdown).")
    min_value: Optional[float] = Field(default=None, description="Minimum value (renders a slider with max_value).")
    max_value: Optional[float] = Field(default=None, description="Maximum value (renders a slider with min_value).")
    step: Optional[float] = Field(default=None, description="Step between selectable values for a slider.")
