import mlflow
from mlflow import MlflowClient


class MLFlowModelRegistry:
    def __init__(self, tracking_uri):
        """Registry to hold and manage multiple models."""
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self._model_cache = {}  # Cache for loaded models

    def register_model(
            self,
            model_identifier: str,
            model,
            info: dict = None,
            tags: dict = None,
    ):
        """ 
            Register a new model in the registry with the given identifier. Tags can be provided in a dictionary. Tags 
            should include the task, e.g. 'prompted_segmentation' or 'instance_segmentation'.
            :param model_identifier: unique identifier for the model
            :param model: model to be registered
            :param info: additional information about the model. Eg. description, name etc.
            :param tags: tags associated with the model. Eg. task: prompted_segmentation, instance_segmentation etc.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        # Store model metadata in MLFlow
        with mlflow.start_run():
            # Log the model using MLFlow
            mlflow.pytorch.log_model(model, model_identifier)
            
            # Log additional info if provided
            if info:
                for key, value in info.items():
                    mlflow.log_param(key, value)
            
            # Set tags if provided
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
    
    def check_registered(self, model_identifier: str):
        """ Check if a model is registered in the registry. """
        try:
            registered_models = self.client.search_registered_models()
            for model in registered_models:
                if model.name == model_identifier:
                    return True
            return False
        except Exception:
            return False
    
    def get_model(self, model_identifier: str):
        """ Get a model from the registry by its identifier. """
        # This method should be cached such that a model is not reinitialized each time.
        if model_identifier in self._model_cache:
            return self._model_cache[model_identifier]
        
        try:
            # Get the latest version of the registered model
            model_uri = f"models:/{model_identifier}/latest"
            model = mlflow.pytorch.load_model(model_uri)
            
            # Cache the model
            self._model_cache[model_identifier] = model
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_identifier}': {str(e)}")
    
    def get_info(self, model_identifier: str):
        """ Get a model info from the registry by its identifier. """
        try:
            registered_model = self.client.get_registered_model(model_identifier)
            return {
                "name": registered_model.name,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
                "description": registered_model.description,
                "tags": registered_model.tags,
            }
        except Exception as e:
            raise ValueError(f"Failed to get info for model '{model_identifier}': {str(e)}")

    def ensure_models_are_registered(self, models: dict[str, dict]):
        """ Ensure that all models are registered in the registry. Registers all unregistered models. """
        for model_id, data in models.items():
            model = data["model"]
            info = data["info"]
            tags = data["tags"]
            if not self.check_registered(model_id):
                self.register_model(model_id, model, info, tags)

