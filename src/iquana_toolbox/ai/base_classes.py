from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import mlflow

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import ModelInfo, InstanceDiscoveryModelInfo, PromptedSegmentationModelInfo, \
    InstanceSegmentationModelInfo
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest, PromptedSegmentationRequest, \
    BaseServiceRequest
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest


class BaseModel(ABC, mlflow.pyfunc.PythonModel):
    """ All models should inherit from this class. It defines the interface that all models must implement. """
    model_info: ModelInfo

    def load_context(self, context):
        """
        Optional: This runs once when the model is loaded.
        Use it to load weights into memory so predict() is fast.
        """
        pass

    @abstractmethod
    def predict(self,
                context: Any,
                model_input: BaseServiceRequest,
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



class BaseInstanceDiscoveryModel(BaseModel):
    """
        Abstract base class for instance discovery models.
        Instance Discovery models take as input an image and an incomplete set of instances, which they have to complete
        or at least predict some new instances. They can additionally take a label as input (e.g. Prompted Concept Segmentation).
    """
    model_info: InstanceDiscoveryModelInfo

    @abstractmethod
    def predict(self, request: InstanceDiscoveryRequest, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """ Process a prompted segmentation request.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        pass


class BasePromptedSegmentationModel(BaseModel):
    """ Abstract base class for 2D prompted segmentation models. """
    model_info: PromptedSegmentationModelInfo

    @abstractmethod
    def predict(self, request: PromptedSegmentationRequest, **kwargs):
        """ Process a prompted segmentation request.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass


class BaseInstanceSegmentationModel(BaseModel):
    """
    Abstract base class for instance segmentation models. Defines the interface that all instance segmentation
    models must implement.
    """
    model_info: InstanceSegmentationModelInfo

    @abstractmethod
    def predict(self, request: InstanceSegmentationRequest, **kwargs) -> list[Contour]:
        """ Inference endpoint. """
        raise NotImplementedError("Subclasses must implement this method!")

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest, **kwargs):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        raise NotImplementedError("Subclasses must implement this method!")
