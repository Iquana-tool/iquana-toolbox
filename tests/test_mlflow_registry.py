import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch


# Ensure local src package import works without installing the project.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from iquana_toolbox.mlflow import MLFlowModelRegistry


class TestMLFlowModelRegistry(unittest.TestCase):
    def test_register_model_creates_version_and_invalidates_cache(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.check_registered = MagicMock(return_value=False)

        run = SimpleNamespace(info=SimpleNamespace(run_id="rid-123"))
        run_context = MagicMock()
        run_context.__enter__.return_value = run
        run_context.__exit__.return_value = False
        mock_log_model = MagicMock()

        registry.client.create_model_version.return_value = SimpleNamespace(version="7")

        with registry._cache_lock:
            registry._model_cache["demo"] = "stale-model"
            registry._model_cache[("demo", "1")] = "stale-model-v1"

        with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri") as mock_set_uri, \
                patch("iquana_toolbox.mlflow.mlflow.start_run", return_value=run_context), \
                patch("iquana_toolbox.mlflow.mlflow.get_artifact_uri", return_value="file:///tmp/model"), \
                patch(
                    "iquana_toolbox.mlflow.mlflow.pytorch",
                    new=SimpleNamespace(log_model=mock_log_model),
                ):
            registry.register_model(
                model_identifier="demo",
                model=object(),
                desc={"description": "segmentation model"},
                tags={"task": "instance_segmentation"},
            )

        mock_set_uri.assert_called_once_with("http://example")
        registry.client.create_registered_model.assert_called_once_with(
            name="demo",
            tags={"task": "instance_segmentation"},
            description="segmentation model",
        )
        mock_log_model.assert_called_once()
        registry.client.create_model_version.assert_called_once_with(
            name="demo",
            source="file:///tmp/model",
            run_id="rid-123",
        )
        with registry._cache_lock:
            self.assertNotIn("demo", registry._model_cache)
            self.assertNotIn(("demo", "1"), registry._model_cache)

    def test_register_model_updates_metadata_when_already_registered(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.check_registered = MagicMock(return_value=True)

        run = SimpleNamespace(info=SimpleNamespace(run_id="rid-9"))
        run_context = MagicMock()
        run_context.__enter__.return_value = run
        run_context.__exit__.return_value = False
        mock_log_model = MagicMock()

        registry.client.create_model_version.return_value = SimpleNamespace(version="1")

        with patch("iquana_toolbox.mlflow.mlflow.set_tracking_uri"), \
                patch("iquana_toolbox.mlflow.mlflow.start_run", return_value=run_context), \
                patch("iquana_toolbox.mlflow.mlflow.get_artifact_uri", return_value="file:///tmp/model"), \
                patch(
                    "iquana_toolbox.mlflow.mlflow.pytorch",
                    new=SimpleNamespace(log_model=mock_log_model),
                ):
            registry.register_model(
                model_identifier="demo",
                model=object(),
                desc="updated description",
                tags={"task": "prompted_segmentation", "team": "cv"},
            )

        registry.client.create_registered_model.assert_not_called()
        registry.client.update_registered_model.assert_called_once_with(
            name="demo",
            description="updated description",
        )
        registry.client.set_registered_model_tag.assert_has_calls(
            [
                call("demo", "task", "prompted_segmentation"),
                call("demo", "team", "cv"),
            ],
            any_order=True,
        )

    def test_ensure_models_are_registered_maps_info_description(self):
        registry = MLFlowModelRegistry("http://example")
        registry.check_registered = MagicMock(return_value=False)
        registry.register_model = MagicMock()

        models = {
            "demo": {
                "model": object(),
                "info": {"description": "from-info"},
                "tags": {"task": "instance_segmentation"},
            }
        }

        registry.ensure_models_are_registered(models)

        registry.register_model.assert_called_once_with(
            "demo",
            models["demo"]["model"],
            "from-info",
            {"task": "instance_segmentation"},
        )

    def test_clone_registered_model_creates_new_entry_and_version(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.check_registered = MagicMock(return_value=False)

        source_model = SimpleNamespace(
            description="source description",
            tags={"task": "instance_segmentation"},
            latest_versions=[
                SimpleNamespace(
                    version="1",
                    source="runs:/run-1/model",
                    run_id="run-1",
                    description="v1",
                    tags={"stage": "old"},
                ),
                SimpleNamespace(
                    version="2",
                    source="runs:/run-2/model",
                    run_id="run-2",
                    description="v2",
                    tags={"stage": "latest"},
                ),
            ],
        )
        registry.client.get_registered_model.return_value = source_model
        registry.client.create_model_version.return_value = SimpleNamespace(version="1")

        with registry._cache_lock:
            registry._model_cache["demo-clone"] = "stale"
            registry._model_cache[("demo-clone", "2")] = "stale-v2"

        result = registry.clone_registered_model("demo-clone", "demo")

        registry.client.create_registered_model.assert_called_once_with(
            name="demo-clone",
            tags={"task": "instance_segmentation"},
            description="source description",
        )
        registry.client.create_model_version.assert_called_once_with(
            name="demo-clone",
            source="runs:/run-2/model",
            run_id="run-2",
        )
        registry.client.update_model_version.assert_called_once_with(
            name="demo-clone",
            version="1",
            description="v2",
        )
        registry.client.set_model_version_tag.assert_called_once_with(
            name="demo-clone",
            version="1",
            key="stage",
            value="latest",
        )
        with registry._cache_lock:
            self.assertNotIn("demo-clone", registry._model_cache)
            self.assertNotIn(("demo-clone", "2"), registry._model_cache)
        self.assertEqual(result["name"], "demo-clone")
        self.assertEqual(result["source_model"], "demo")
        self.assertEqual(result["source_version"], "2")

    def test_clone_registered_model_raises_when_source_has_no_versions(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.client.get_registered_model.return_value = SimpleNamespace(
            latest_versions=[],
            description=None,
            tags=None,
        )

        with self.assertRaises(ValueError):
            registry.clone_registered_model("demo-clone", "demo")

    def test_get_model_by_alias_raises_on_lookup_error(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.client.get_model_version_by_alias.side_effect = RuntimeError("not found")

        with self.assertRaises(ValueError):
            registry.get_model_by_alias("demo", "prod")

    def test_get_model_by_alias_latest_resolves_highest_version(self):
        registry = MLFlowModelRegistry("http://example")
        registry.client = MagicMock()
        registry.client.get_registered_model.return_value = SimpleNamespace(
            latest_versions=[
                SimpleNamespace(version="1"),
                SimpleNamespace(version="7"),
                SimpleNamespace(version="3"),
            ]
        )
        registry.get_model_by_version = MagicMock(return_value="model-obj")

        result = registry.get_model_by_alias("demo", "latest")

        self.assertEqual(result, "model-obj")
        registry.get_model_by_version.assert_called_once_with("demo", "7")


if __name__ == "__main__":
    unittest.main()



