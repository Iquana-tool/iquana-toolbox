import logging
import threading
from pathlib import Path

import mlflow
from cachetools import TTLCache
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from iquana_toolbox.ai.base_classes import BaseModel
from iquana_toolbox.schemas.model_info import parse_tags_to_model_info, ModelInfo

logger = logging.getLogger(__name__)


class MLFlowModelRegistry:
    def __init__(self, tracking_uri, cache_maxsize: int = 8, cache_ttl_seconds: int = 3600):
        """Registry to hold and manage multiple models."""
        self.tracking_uri = tracking_uri
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self._model_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl_seconds)
        self._cache_lock = threading.Lock()

    def _invalidate_model_cache(self, model_identifier: str):
        """Clear all cached entries for a model across versions/aliases."""
        with self._cache_lock:
            keys_to_remove = []
            for key in list(self._model_cache.keys()):
                if key == model_identifier:
                    keys_to_remove.append(key)
                elif isinstance(key, tuple) and key and key[0] == model_identifier:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                self._model_cache.pop(key, None)

    @staticmethod
    def _normalize_model_source_uri(source: str) -> str:
        """Normalize local filesystem source paths to file:// URIs for MLflow compatibility."""
        if not source:
            return source
        lower_source = source.lower()
        if lower_source.startswith(("runs:/", "models:/", "file:", "http:", "https:", "s3:", "gs:", "mlflow-artifacts:")):
            return source
        if len(source) >= 3 and source[1] == ":" and source[2] in ("\\", "/"):
            return Path(source).resolve().as_uri()
        return source

    def register_model(
            self,
            model: BaseModel,
    ):
        """ 
            Register a new model in the registry with the given identifier. Tags can be provided in a dictionary. Tags 
            should include the task, e.g. 'prompted_segmentation' or 'instance_segmentation'.
            :param model: model to be registered
        """

        mlflow.set_tracking_uri(self.tracking_uri)

        if not self.check_registered(model.model_info.registry_key):
            self.client.create_registered_model(
                name=model.model_info.registry_key,
                tags=model.model_info.model_dump(),
                description=model.model_info.description,
            )
            logger.info(
                f"Registered model {model.model_info.name} in MLflow with key {model.model_info.registry_key}.")
        else:
            logger.info(f"Model {model.model_info.name} is already registered in MLflow. Skipped.")

        with mlflow.start_run() as run:
            info = mlflow.pyfunc.log_model(
                python_model=model,
                registered_model_name=model.model_info.registry_key,
                metadata=model.model_info.model_dump()
            )

        self._invalidate_model_cache(model.model_info.registry_key)
    
    def check_registered(self, model_identifier: str):
        """ Check if a model is registered in the registry. """
        try:
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

    def _get_model_cached(self, version: ModelVersion):
        with self._cache_lock:
            model = self._model_cache.get((version.name, version.version))
            if model is not None:
                logger.debug(f"Cache hit for model '{version.name}-{version.version}'")
                return model
            else:
                logger.info(f"Cache miss for model '{version.name}-{version.version}'. Loading from MLflow.")
                mlflow.set_tracking_uri(self.tracking_uri)
                model = mlflow.pytorch.load_model(version.source)
                self._model_cache[(version.name, version.version)] = model
                return model

    def get_model_by_version(self, model_identifier: str, version: str):
        """ Get a model from the registry by its identifier and version. """
        version = self.client.get_model_version(name=model_identifier, version=version)
        return self._get_model_cached(version)

    def get_model_by_alias(self, model_identifier: str, alias: str):
        """ Get a model from the registry by its identifier and alias. """
        version = self.client.get_model_version_by_alias(name=model_identifier, alias=alias)
        return self._get_model_cached(version)

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
    
    def get_model_info(self, model_identifier: str) -> ModelInfo:
        """ Get a model info from the registry by its identifier. """
        try:
            logger.debug("Fetching model info for '%s'.", model_identifier)
            registered_model = self.client.get_registered_model(model_identifier)
            model_info = parse_tags_to_model_info(registered_model.tags)
            return model_info
        except Exception as e:
            logger.exception("Failed to fetch model info for '%s'.", model_identifier)
            raise ValueError(f"Failed to get info for model '{model_identifier}': {str(e)}")

    def ensure_models_are_registered(self, models: list):
        """ Ensure that all models are registered in the registry. Registers all unregistered models. """
        for model_cls in models:
            if not self.check_registered(model_cls.model_info.registry_key):
                self.register_model(model_cls())
            else:
                continue

