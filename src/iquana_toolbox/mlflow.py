import logging
import threading

import mlflow
from cachetools import TTLCache
from mlflow import MlflowClient


logger = logging.getLogger(__name__)


class MLFlowModelRegistry:
    def __init__(self, tracking_uri, cache_maxsize: int = 8, cache_ttl_seconds: int = 3600):
        """Registry to hold and manage multiple models."""
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self._model_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl_seconds)
        self._cache_lock = threading.Lock()

    def register_model(
            self,
            model_identifier: str,
            model,
            desc: str | dict | None = None,
            tags: dict = None,
    ):
        """ 
            Register a new model in the registry with the given identifier. Tags can be provided in a dictionary. Tags 
            should include the task, e.g. 'prompted_segmentation' or 'instance_segmentation'.
            :param model_identifier: unique identifier for the model
            :param model: model to be registered
            :param desc: Description of the model
            :param tags: tags associated with the model. Eg. task: prompted_segmentation, instance_segmentation etc.
        """
        logger.info("Registering model '%s' in MLflow.", model_identifier)
        mlflow.set_tracking_uri(self.tracking_uri)

        # Backward compatibility: callers may still pass an info dict as `desc`.
        if isinstance(desc, dict):
            desc = desc.get("description") or desc.get("desc")

        if not self.check_registered(model_identifier):
            self.client.create_registered_model(
                name=model_identifier,
                tags=tags,
                description=desc,
            )
            logger.info("Created MLflow registered model '%s'.", model_identifier)
        else:
            if desc:
                self.client.update_registered_model(name=model_identifier, description=desc)
            if tags:
                for key, value in tags.items():
                    self.client.set_registered_model_tag(model_identifier, key, value)

        with mlflow.start_run() as run:
            artifact_path = "model"
            mlflow.pytorch.log_model(pytorch_model=model, registered_model_name=model_identifier)
            source = f"runs:/{run.info.run_id}/{artifact_path}"
            model_version = self.client.create_model_version(
                name=model_identifier,
                source=source,
                run_id=run.info.run_id,
            )

        with self._cache_lock:
            self._model_cache.pop(model_identifier, None)

        logger.info(
            "Model '%s' registered as version '%s'.",
            model_identifier,
            model_version.version,
        )
    
    def check_registered(self, model_identifier: str):
        """ Check if a model is registered in the registry. """
        try:
            logger.debug("Checking registration status for model '%s'.", model_identifier)
            registered_models = self.client.search_registered_models()
            for model in registered_models:
                if model.name == model_identifier:
                    logger.debug("Model '%s' is registered.", model_identifier)
                    return True
            logger.debug("Model '%s' is not registered.", model_identifier)
            return False
        except Exception:
            logger.exception("Failed to check registration status for model '%s'.", model_identifier)
            return False

    def clone_registered_model(self, new_identifier: str, old_identifier: str):
        """
            Clone a model and make a new registry entry. Useful for tracking user specific models.
        """
        # TODO: Implement this
    
    def get_model(self, model_identifier: str, version_or_alias: str = 'latest'):
        """ Get a model from the registry by its identifier. """
        with self._cache_lock:
            model = self._model_cache.get(model_identifier)
            if model is not None:
                logger.debug("Cache hit for model '%s'.", model_identifier)
                return model

            try:
                logger.info("Cache miss for model '%s'. Loading from MLflow.", model_identifier)
                mlflow.set_tracking_uri(self.tracking_uri)
                model_uri = f"models:/{model_identifier}/{version_or_alias}"
                model = mlflow.pytorch.load_model(model_uri)
                self._model_cache[model_identifier] = model
                logger.info("Loaded and cached model '%s' from '%s'.", model_identifier, model_uri)
                return model
            except Exception as e:
                logger.exception("Failed to load model '%s' from MLflow.", model_identifier)
                raise ValueError(f"Failed to load model '{model_identifier}': {str(e)}")

    @staticmethod
    def _build_tag_filter(tags: dict):
        """ Build an MLflow filter string for searching models by tags. """
        filters = []
        for key, value in tags.items():
            filters.append(f"tags.{key} = '{value}'")
        return " AND ".join(filters)

    def get_models_via_tags(self, tags: dict):
        """ Get models from the registry that match the given tags. """
        try:
            logger.debug("Searching for models with tags: %s", tags)
            registered_models = self.client.search_registered_models(filter_string=self._build_tag_filter(tags))
            model_infos = []
            for model in registered_models:
                model_infos.append({
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "tags": model.tags,
                    "versions": model.latest_versions
                })
            logger.debug("Found %d models matching tags.", len(model_infos))
            return model_infos
        except Exception as e:
            logger.exception("Failed to search for models with tags: %s", tags)
            raise ValueError(f"Failed to search for models with tags {tags}: {str(e)}")
    
    def get_model_info(self, model_identifier: str):
        """ Get a model info from the registry by its identifier. """
        try:
            logger.debug("Fetching model info for '%s'.", model_identifier)
            registered_model = self.client.get_registered_model(model_identifier)
            return {
                "name": registered_model.name,
                "creation_timestamp": registered_model.creation_timestamp,
                "last_updated_timestamp": registered_model.last_updated_timestamp,
                "description": registered_model.description,
                "tags": registered_model.tags,
            }
        except Exception as e:
            logger.exception("Failed to fetch model info for '%s'.", model_identifier)
            raise ValueError(f"Failed to get info for model '{model_identifier}': {str(e)}")

    def ensure_models_are_registered(self, models: dict[str, dict]):
        """ Ensure that all models are registered in the registry. Registers all unregistered models. """
        for model_id, data in models.items():
            model = data["model"]
            info = data.get("info")
            desc = data.get("desc")
            if desc is None and isinstance(info, str):
                desc = info
            if desc is None and isinstance(info, dict):
                desc = info.get("description") or info.get("desc")
            tags = data.get("tags")
            if not self.check_registered(model_id):
                logger.info("Model '%s' is missing in registry. Registering now.", model_id)
                self.register_model(model_id, model, desc, tags)
            else:
                logger.debug("Model '%s' already registered; skipping.", model_id)

