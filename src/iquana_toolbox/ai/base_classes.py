from abc import ABC, abstractmethod

import numpy as np

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest, PromptedSegmentationRequest
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest


class BaseInstanceDiscoveryModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def inference(self, request: InstanceDiscoveryRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process a prompted segmentation request.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        pass



class BasePromptedSegmentationModel(ABC):
    """ Abstract base class for 2D prompted segmentation models. """
    @abstractmethod
    def inference(self, request: PromptedSegmentationRequest):
        """ Process a prompted segmentation request.
        :param request: The request to be processed.
        :return: A tuple containing a mask and their corresponding quality score.
        """
        pass


class BaseInstanceSegmentationModel(ABC):
    """
    Abstract base class for instance segmentation models. Defines the interface that all instance segmentation
    models must implement.
    """
    @abstractmethod
    def inference(self, request: InstanceSegmentationRequest) -> list[Contour]:
        """ Inference endpoint. """
        raise NotImplementedError("Subclasses must implement this method!")

    @abstractmethod
    def train(self, request: InstanceSegmentationTrainingRequest):
        """
            Send a job to celery to train this model. The training request contains all the necessary information to
            train the model, including the dataset, the label hierarchy, and the training configuration.
        """
        raise NotImplementedError("Subclasses must implement this method!")
