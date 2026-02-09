from collections import defaultdict
from typing import Optional, Literal
from typing import Tuple
from pydantic import BaseModel, Field, computed_field

from iquana_toolbox.schemas.labels import LabelHierarchy


class HyperParams(BaseModel):
    batch_size: int = Field(default=32, description="Batch size to use for training.")
    learning_rate: float = Field(default=0.001, description="Learning rate to use for training.")
    early_stopping_patience: int = Field(
        default=25,
        description="Number of epochs to wait for improvement. If this number is exceeded without the model improving, "
                    "training is stopped regardless of the amount of remaining epochs. If this value is <= 0, "
                    "no early stopping will be applied."
    )


class Augmentations(BaseModel):
    """ Schema for augmentations. Augmentations are any image operations that should be applied before an image is fed to
        the model. This can include cropping etc, but also normalizing.
    """
    crop_relative_min: Optional[float] = Field(
        default=0.25,
        description="Minimum relative size for random crop (e.g., 0.25 = up to 25% of image size)."
    )
    rotation_degrees: Optional[int] = Field(
        default=None,
        description="Degrees for random rotation. If None, no rotation is applied."
    )
    use_horizontal_flip: bool = Field(
        default=True,
        description="Whether to use horizontal flip of images."
    )
    use_vertical_flip: bool = Field(
        default=True,
        description="Whether to use vertical flip of images."
    )
    color_jitter: Optional[Tuple[float, float, float, float]] | None = Field(
        default=None,
        description="Color jitter parameters (Brightness, Contrast, Saturation, Hue). If None, not applied."
    )


class SemanticTrainingData(BaseModel):
    image_urls: list[str] = Field(..., description="List of image urls to train on.")
    mask_urls: list[str] = Field(..., description="List of mask urls to train on.")
    label_hierarchy: LabelHierarchy = Field(
        description="Label hierarchy to be used for training and evaluation."
    )


class SemanticTrainingConfig(BaseModel):
    val_ratio: float = Field(default=0.1, description="Ratio of training data to validation data.")
    image_size: tuple = Field((224, 224), description="Image size.")
    loss: Literal["cross_entropy", "dice_loss", "focal_loss"] = Field(
        default="cross_entropy",
        description="The loss the model should be trained on. This is currently hardcoded to three options: "
                    "cross_entropy, dice_loss and focal_loss."
    )
    num_epochs: int = Field(default=100, description="Number of epochs to train.")

    # Hyperparameters; Not sure if we should concern users with this kind of stuff. However, allowing more control
    # can only be better.
    hyper_params: Optional[HyperParams] = Field(default_factory=HyperParams, description="Hyperparameters")
    augmentations: Optional[Augmentations] = Field(default_factory=Augmentations, description="Augmentations")

class SemanticTrainingRequest(SemanticTrainingData, SemanticTrainingConfig):
    model_registry_key: str = Field(default="unet", description="A key from the model registry")


class Metrics(BaseModel):
    """ Tracks an arbitrary number of metrics per epoch."""
    metrics: dict[str, list] = Field(
        default=defaultdict(list),
        description="Dictionary of metrics such as loss, accuracy, etc. Maps the metric name to its values per epoch."
    )

    def add_metric(self, name, value):
        self.metrics[name].append(value)

    def add_metrics(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            self.add_metric(k, v)

    def get_epoch_metrics(self, epoch):
        return {k: v[epoch] for k, v in self.metrics.items()}

    def __getattr__(self, item):
        return self.metrics[item]

    def __len__(self):
        return len(list(self.metrics.values())[0])


class TrainingProgress(BaseModel):
    status: Literal["PROGRESS", "STOPPED", "SUCCESS", "FAILED"] = Field(
        default="PROGRESS",
    )
    # Epoch fields
    epoch_count: int = Field(default=0, description="The total number of epochs trained.")
    best_epoch: int = Field(default=-1, description="Best epoch.")

    # Monitored metric: The metric that is monitored to identify the best epoch. For example the loss of the validation set
    monitored_metric: str = Field(
        default="loss",
        description="The metric that is monitored to identify the best epoch. For example the dice score."
    )
    monitored_metric_type: Literal["train", "val"] = Field(
        default="train",
        description="Whether to monitor train or validation values for figuring out best epoch."
    )
    monitored_metric_lower_is_better: bool = Field(
        default=True,
        description="Whether the metric gets better with decreasing (True) or increasing (False) values."
    )
    monitored_metric_best_value: float | None = Field(default=None, description="Value of the best epoch metric.")

    # Tracking metrics
    train_metrics: Metrics = Field(default_factory=Metrics,
                                   description="List of training metrics. Tracks metrics over epochs.")
    val_metrics: Metrics = Field(default_factory=Metrics,
                                 description="List of validation metrics. Tracks metrics over epochs.")

    @computed_field
    @property
    def performance(self) -> dict[str, dict[str, float]]:
        return {
            "train": self.train_metrics.get_epoch_metrics(self.best_epoch),
            "val": self.val_metrics.get_epoch_metrics(self.best_epoch)
        }

    def training_step(self, train_metrics, val_metrics=None) -> bool:
        """ Update training step information. Returns true if the new epoch was better than the best epoch yet,
        else false. """
        self.progress.train_metrics.add_metrics(train_metrics)
        if val_metrics is not None:
            self.progress.val_metrics.add_metrics(val_metrics)
            if self.monitor_type == "val":
                raise ValueError("Best epoch determination needs val metrics, but none were given.")

        is_new_best = self.is_new_best_epoch()
        self.epoch_count += 1
        return is_new_best

    @property
    def monitored_metric_latest_value(self):
        if self.monitored_metric_type == "val":
            return self.val_metrics[self.monitored_metric][-1]
        else:
            return self.train_metrics[self.monitored_metric][-1]

    def add_test_metrics(self, test_metrics):
        difference = len(self.train_metrics) - len(self.test_metrics)
        for _ in range(difference):
            empty_metrics = {k: None for k in test_metrics.keys()}
            self.test_metrics.add_metrics(empty_metrics)
        self.test_metrics.add_metrics(test_metrics)

    def is_new_best_epoch(self):
        # Check the condition for updating the best epoch, aka is it greater or smaller than the best seen value
        condition = (
                (self.monitored_metric_lower_is_better and self.monitored_metric_latest_value < self.monitored_metric_best_value)
                or
                (not self.monitored_metric_lower_is_better and self.monitored_metric_latest_value > self.monitored_metric_best_value)
        )
        # Update if it is better or if we have not yet recorded a value.
        if self.monitored_metric_best_value is None or condition:
            self.monitored_metric_best_value = self.monitored_metric_latest_value
            self.best_epoch = self.epoch_count
            return True
        else:
            return False
