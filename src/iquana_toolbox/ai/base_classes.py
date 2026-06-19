from abc import ABC, abstractmethod
from typing import Any

import mlflow
import numpy as np

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import ModelInfo, InstanceDiscoveryModelInfo, PromptedSegmentationModelInfo, \
    InstanceSegmentationModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest, PromptedSegmentationRequest, \
    BaseServiceRequest
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest


class BaseModel(ABC, mlflow.pyfunc.PythonModel):
    """
    All models should inherit from this class. It defines the interface that all models must implement.

    Attributes:
        model_info: ModelInfo: A dataclass containing metadata about the model, such as its name, description, and
            registry key. Check out the docs for ModelInfo for more information.
        default_hyperparameters: dict[str, Any]: A dictionary containing the default hyperparameters for training the
            model. This must not necessarily be overwritten by subclasses.
    """
    model_info: ModelInfo
    default_hyperparameters: None | dict[str, Any] = None

    # Names of instance attributes that must NOT be cloudpickled when the model is
    # logged (e.g. live ``torch.nn.Module`` / HuggingFace objects). MLflow serializes
    # the whole python_model with cloudpickle; recent ``transformers`` versions attach
    # ``ContextVar``-backed forward hooks that cloudpickle cannot serialize, so these
    # attributes are stripped from ``__getstate__`` and must be rebuilt in
    # ``load_context`` (from ``self.get_artifacts`` output or from the original source).
    _unpicklable_attrs: tuple[str, ...] = ()

    def get_artifacts(self, tmp_dir: str) -> dict[str, str] | None:
        """
        Optional: persist heavy weights to ``tmp_dir`` and return an
        ``{artifact_name: local_path}`` mapping. The registry passes this to
        ``mlflow.pyfunc.log_model(artifacts=...)``; at load time the paths are
        exposed via ``context.artifacts[artifact_name]`` to ``load_context``.

        Return ``None`` (default) for models whose weights can simply be reloaded
        from their original source (e.g. a HuggingFace Hub id) inside ``load_context``.
        """
        return None

    def __getstate__(self):
        """Drop un-pickleable live model attributes; they are rebuilt in load_context."""
        state = self.__dict__.copy()
        for attr in self._unpicklable_attrs:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def load_context(self, context):
        """
        Optional: This runs once when the model is loaded.
        Use it to load weights into memory so predict() is fast.
        """
        pass

    @abstractmethod
    def predict(self,
                context: Any,
                model_input: list[BaseServiceRequest],
                params: dict[str, Any] | None = None):
        raise NotImplementedError("Subclasses must implement this method!")

    @abstractmethod
    def train(self,
              request: InstanceSegmentationTrainingRequest,
              **kwargs):
        """
            Model-specific training method. Each trainable model implements this for itself, such that the training
            endpoint can call it without knowing the details of the model. The training request contains all the
            necessary information to train the model such as hyperparameters.
        """
        raise NotImplementedError("Subclasses must implement this method!")



class InstanceDiscoveryModel(BaseModel):
    """
        Abstract base class for instance discovery models.
        Instance Discovery models take as input an image and an incomplete set of instances, which they have to complete
        or at least predict some new instances. They can additionally take a label as input (e.g. Prompted Concept Segmentation).
    """
    model_info: InstanceDiscoveryModelInfo

    @abstractmethod
    def predict(self,
                context: Any,
                model_input: list[InstanceDiscoveryRequest],
                params: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Process an InstanceDiscoveryRequest.
        """
        pass

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        pass


class PromptedSegmentationModel(BaseModel):
    """ Abstract base class for 2D prompted segmentation models. """
    model_info: PromptedSegmentationModelInfo

    @abstractmethod
    def predict(self,
                context: Any,
                model_input: list[PromptedSegmentationRequest],
                params: dict[str, Any] | None = None):
        """ Process a prompted segmentation request."""
        pass


class InstanceSegmentationModel(BaseModel):
    """
    Abstract base class for instance segmentation models. Defines the interface that all instance segmentation
    models must implement.
    """
    model_info: InstanceSegmentationModelInfo

    @abstractmethod
    def predict(self,
                context: Any,
                model_input: list[InstanceSegmentationRequest],
                params: dict[str, Any] | None = None) -> list[Contour]:
        """ Inference endpoint. """
        raise NotImplementedError("Subclasses must implement this method!")

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        raise NotImplementedError("Subclasses must implement this method!")
