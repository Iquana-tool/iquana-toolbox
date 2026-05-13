from typing import Union, Any, Optional

from pydantic import BaseModel, Field, field_validator

from iquana_toolbox.schemas.database.labels import Label


class BaseTrainingRequest(BaseModel):
    dataset_id: int = Field(..., title="Dataset ID", description="The dataset ID.")
    image_folder_path: str = Field(..., description="Path to the folder containing the images.")
    model_registry_key: str = Field(..., description="A key from the model registry")
    user_id: Union[str, int] = Field(..., title="User ID", description="Unique identifier for the user.")
    hyper_parameter: dict = Field(default_factory=dict, description="Hyperparameters of the training.")


class InstanceSegmentationTrainingRequest(BaseTrainingRequest):
    label: Label = Field(..., title="Label", description="Label defining the instances.")
    annotation_file_url: str = Field(... , title="Annotation File URL",
                                     description="The path to a COCO annotation file.")


class HyperParameter(BaseModel):
    """
    Represents a hyperparameter for training. This will be used to generate a UI for selecting hyperparameters.
    """
    key: str = Field(..., title="Hyperparameter", description="The key of the hyperparameter. This will be passed to "
                                                              "training scripts.")
    value: str = Field(..., title="Hyperparameter", description="The value of the hyperparameter. This will be passed to "
                                                                "training scripts.")
    default_value: Any = Field(..., title="Default value", description="Default hyperparameter value.")
    description: Any = Field(..., title="Description", description="Description of the hyperparameter.")


class HyperParameterSelector(HyperParameter):
    """
    Represents a hyperparameter for training. This will be used to generate a UI for selecting hyperparameters.
    This class implements hyperparameters that have discrete values to choose from. E.g. a lr scheduler or the loss.
    """
    options: list[Any] = Field(
        default_factory=list,
        description="A list of possible values to select from. E.g. LR schedulers"
    )


class HyperparameterRangeSelector(HyperParameter):
    """
        Represents a hyperparameter for training. This will be used to generate a UI for selecting hyperparameters.
        This class implements hyperparameters that have a range of values to choose from. E.g. learning rate.
    """
    min_value: Any = Field(..., title="Minimum value", description="Minimum hyperparameter value.")
    max_value: Any = Field(..., title="Maximum value", description="Maximum hyperparameter value.")
    step: Any = Field(..., title="Step value", description="Step hyperparameter value.")
