import logging
import threading
from pathlib import Path

import mlflow
from cachetools import TTLCache
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

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
            mlflow.pytorch.log_model(pytorch_model=model, name=artifact_path)
            source = mlflow.get_artifact_uri(artifact_path)
            source = self._normalize_model_source_uri(source)
            model_version = self.client.create_model_version(
                name=model_identifier,
                source=source,
                run_id=run.info.run_id,
            )

        self._invalidate_model_cache(model_identifier)

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
        if new_identifier == old_identifier:
            raise ValueError("new_identifier must be different from old_identifier")

        logger.info(
            "Cloning registered model from '%s' to '%s'.",
            old_identifier,
            new_identifier,
        )

        try:
            source_model = self.client.get_registered_model(old_identifier)
            latest_versions = list(source_model.latest_versions or [])
            if not latest_versions:
                raise ValueError(
                    f"Source model '{old_identifier}' has no versions to clone."
                )

            source_version = max(latest_versions, key=lambda version: int(version.version))

            if not self.check_registered(new_identifier):
                self.client.create_registered_model(
                    name=new_identifier,
                    tags=source_model.tags,
                    description=source_model.description,
                )
                logger.info("Created destination registered model '%s'.", new_identifier)
            else:
                if source_model.description:
                    self.client.update_registered_model(
                        name=new_identifier,
                        description=source_model.description,
                    )
                if source_model.tags:
                    for key, value in source_model.tags.items():
                        self.client.set_registered_model_tag(new_identifier, key, value)

            cloned_version = self.client.create_model_version(
                name=new_identifier,
                source=self._normalize_model_source_uri(source_version.source),
                run_id=source_version.run_id,
            )

            if getattr(source_version, "description", None):
                self.client.update_model_version(
                    name=new_identifier,
                    version=cloned_version.version,
                    description=source_version.description,
                )

            source_version_tags = getattr(source_version, "tags", None) or {}
            for key, value in source_version_tags.items():
                self.client.set_model_version_tag(
                    name=new_identifier,
                    version=cloned_version.version,
                    key=key,
                    value=value,
                )

            self._invalidate_model_cache(new_identifier)

            logger.info(
                "Cloned model '%s' version '%s' into '%s' version '%s'.",
                old_identifier,
                source_version.version,
                new_identifier,
                cloned_version.version,
            )
            return {
                "name": new_identifier,
                "version": cloned_version.version,
                "source_model": old_identifier,
                "source_version": source_version.version,
            }
        except ValueError:
            raise
        except Exception as e:
            logger.exception(
                "Failed to clone registered model from '%s' to '%s'.",
                old_identifier,
                new_identifier,
            )
            raise ValueError(
                f"Failed to clone registered model '{old_identifier}' to '{new_identifier}': {str(e)}"
            )

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

