from abc import ABC, abstractmethod

import numpy as np
import mlflow

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest, PromptedSegmentationRequest
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest


class BaseModel(ABC, mlflow.pyfunc.PythonModel):
    pass

class BaseInstanceDiscoveryModel(BaseModel):
    """ Abstract base class for 2D prompted segmentation models. """
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
