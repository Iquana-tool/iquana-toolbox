import inspect
import json
import logging
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import mlflow
from cachetools import TTLCache
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel

from iquana_toolbox.ai.base_classes import BaseModel
from iquana_toolbox.schemas.model_info import parse_tags_to_model_info, ModelInfo

logger = logging.getLogger(__name__)


def _infer_code_paths(model) -> Optional[list[str]]:
    """Best-effort source dir(s) to bundle with a logged model.

    ``mlflow.pyfunc.log_model(python_model=<instance>)`` cloudpickles the model by
    *reference* to its defining module (e.g. ``models.mask2former``). When the model
    is loaded in a different process (a Celery worker) whose ``sys.path`` does not
    include the service root, that import fails with ``No module named 'models'``.

    Bundling the model's top-level package via ``code_paths`` makes the artifact
    self-contained: MLflow copies it under the artifact's ``code/`` dir and puts that
    on ``sys.path`` at load time. Returns ``None`` when nothing sensible can be
    inferred (e.g. classes defined in ``__main__``), leaving logging unchanged.
    """
    try:
        module = inspect.getmodule(type(model))
        if module is None:
            return None
        top_name = (module.__name__ or "").split(".")[0]
        if top_name in ("", "__main__", "builtins"):
            return None
        top_pkg = sys.modules.get(top_name)
        if top_pkg is not None and getattr(top_pkg, "__path__", None):
            return [str(Path(list(top_pkg.__path__)[0]).resolve())]
        module_file = getattr(module, "__file__", None)
        return [str(Path(module_file).resolve())] if module_file else None
    except Exception:
        logger.debug("Could not infer code_paths for %r", type(model), exc_info=True)
        return None


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
                elif isinstance(key, str) and (
                        key.startswith(f"models:/{model_identifier}/")
                        or key.startswith(f"models:/{model_identifier}@")
                ):
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
        if lower_source.startswith(
                ("runs:/", "models:/", "file:", "http:", "https:", "s3:", "gs:", "mlflow-artifacts:")):
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

        model_is_registered = self.check_registered(model.model_info.registry_key)
        if ((not model_is_registered)
                or "user_id" in model.model_info.tags or "dataset_id" in model.model_info.tags):
            # If the model is not registered or this is a trained model (user_id or dataset_id is present), we log a new
            # version of the model.
            # Models may declare heavy/un-pickleable weights via ``get_artifacts``; we save
            # them to a temp dir and hand them to MLflow as artifacts (resolved back to local
            # paths in ``load_context``) instead of cloudpickling the live objects.
            artifacts_dir = tempfile.mkdtemp(prefix="iquana_model_artifacts_")
            try:
                artifacts = model.get_artifacts(artifacts_dir)
                # Bundle the model's source so it loads in workers that don't have the
                # service root on sys.path (avoids "No module named 'models'" at load).
                code_paths = _infer_code_paths(model)
                model_info_tags = self._model_info_tags(model.model_info)
                with mlflow.start_run():
                    mlflow.pyfunc.log_model(
                        python_model=model,
                        registered_model_name=model.model_info.registry_key,
                        tags=model_info_tags,
                        metadata=model.model_info.model_dump(mode="json"),
                        artifacts=artifacts,
                        code_paths=code_paths,
                    )
            finally:
                shutil.rmtree(artifacts_dir, ignore_errors=True)
            logger.info(
                f"Registered model {model.model_info.name} in MLflow with key {model.model_info.registry_key}.")
        else:
            logger.info(f"Model {model.model_info.name} is an already registered base model. "
                        f"Skipped re-logging; syncing its tags.")

        # Always synced, on both paths, because tags are *metadata about the code* -- not
        # about the weights. A base model's artifact is skipped above (its weights have not
        # changed), but the class it points at may well have: adding a capability mixin to a
        # model changes which tasks it advertises, and the discovery layer filters on exactly
        # these tags. Syncing only when re-logging left such a model permanently invisible to
        # the surface it had just gained -- and since cloudpickle stores importable classes by
        # *reference*, the loaded object already had the new capability; only the tags lied.
        #
        # This is also the mechanism MLflow 3.x forces on us regardless: ``log_model(tags=...)``
        # attaches them to the LoggedModel entity, while our read path
        # (get_model_info / get_model_infos_via_tags) queries the *registered model's* tags.
        model_info_tags = self._model_info_tags(model.model_info)
        for key, value in model_info_tags.items():
            self.client.set_registered_model_tag(
                model.model_info.registry_key, key, str(value)
            )
        if model_is_registered and not model.model_info.input_contracts:
            # Keep an explicit empty value authoritative over stale contracts in
            # the artifact metadata. Removing the tag would make discovery fall
            # back to that stale metadata again.
            self.client.set_registered_model_tag(
                model.model_info.registry_key, "input_contracts", "[]"
            )
        # ``log_model`` only stores the description inside the artifact metadata, but
        # consumers read ``RegisteredModel.description`` directly. Mirror it across.
        if model.model_info.description:
            self.client.update_registered_model(
                model.model_info.registry_key,
                description=model.model_info.description,
            )

        self._invalidate_model_cache(model.model_info.registry_key)

    @staticmethod
    def _model_info_tags(model_info: ModelInfo) -> dict[str, str]:
        """Return discovery tags, including typed inference contracts.

        MLflow registered-model tags are string-valued.  The full typed
        ``ModelInfo`` is also stored in logged-model metadata, but the toolbox
        registry's discovery path reads registered-model tags.  Serialize only
        the contract field here so legacy tags keep their existing shape while
        contract-aware discovery remains lossless.
        """
        tags = {str(key): str(value) for key, value in model_info.tags.items()}
        tags.pop("input_contracts", None)
        if model_info.input_contracts:
            tags["input_contracts"] = json.dumps(
                [contract.model_dump(mode="json") for contract in model_info.input_contracts],
                separators=(",", ":"),
                sort_keys=True,
            )
        return tags

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

    def _get_model_cached(self, model_uri: str) -> PyFuncModel:
        with self._cache_lock:
            model = self._model_cache.get(model_uri)
            if model is not None:
                logger.debug(f"Cache hit for model '{model_uri}'")
                return model
            else:
                logger.info(f"Cache miss for model '{model_uri}'. Loading from MLflow.")
                mlflow.set_tracking_uri(self.tracking_uri)
                model = mlflow.pyfunc.load_model(model_uri)
                self._model_cache[model_uri] = model
                return model

    @staticmethod
    def _build_model_uri(model_identifier: str, *, version: Optional[str] = None, alias: Optional[str] = None) -> str:
        """Build an MLflow models:/ URI from either a version or alias."""
        if bool(version) == bool(alias):
            raise ValueError("Exactly one of 'version' or 'alias' must be provided.")
        if version:
            return f"models:/{model_identifier}/{version}"
        return f"models:/{model_identifier}@{alias}"

    def get_model_by_version(self, model_identifier: str, version: str) -> PyFuncModel:
        """ Get a model from the registry by its identifier and version. """
        model_uri = self._build_model_uri(model_identifier, version=version)
        return self._get_model_cached(model_uri)

    def get_model_by_alias(self, model_identifier: str, alias: str) -> PyFuncModel:
        """ Get a model from the registry by its identifier and alias. """
        model_uri = self._build_model_uri(model_identifier, alias=alias)
        return self._get_model_cached(model_uri)

    @staticmethod
    def _build_tag_filter(tags: dict):
        """ Build an MLflow filter string for searching models by tags. """
        filters = []
        for key, value in tags.items():
            filters.append(f"tags.{key} = '{value}'")
        return " AND ".join(filters)

    def get_model_infos_via_tags(self, tags: dict) -> list[ModelInfo]:
        """ Get models from the registry that match the given tags. """
        try:
            logger.debug("Searching for models with tags: %s", tags)
            registered_models = self.client.search_registered_models(filter_string=self._build_tag_filter(tags))
            model_infos = []
            for model in registered_models:
                model_infos.append(self.get_model_info(model.name))
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

    def register_models(self, models: list):
        """ Ensure that all models are registered in the registry.
        Registers all unregistered models. """
        for model_cls in models:
            self.register_model(model_cls())
