"""Tests for the quantification metric registry and the geometry metrics.

Ensures the registry registers the four geometry metrics, rejects duplicate keys,
produces a serializable catalog, and — crucially — that the registry metrics agree
exactly with :meth:`QuantificationModel.from_contour` on the same contour (they must
share the underlying geometry math, not duplicate it).
"""
import numpy as np
import pytest

from iquana_toolbox.quantification import (
    METRIC_REGISTRY,
    Metric,
    QuantContext,
    Tier,
    UnitKind,
    get_metric,
    list_metrics,
    register_metric,
    resolve_unit,
)
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.quantification import QuantificationModel

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 500


def make_rectangle_contour(contour_id=1):
    """A 100 x 50 px rectangle on a 1000 x 500 image, stored normalized."""
    x_px = [100, 200, 200, 100]
    y_px = [100, 100, 150, 150]
    return Contour(
        id=contour_id,
        x=[x / IMAGE_WIDTH for x in x_px],
        y=[y / IMAGE_HEIGHT for y in y_px],
    )


class TestRegistry:
    def test_geometry_metrics_registered(self):
        for key in ("area", "perimeter", "circularity", "max_diameter"):
            assert key in METRIC_REGISTRY
            assert isinstance(get_metric(key), Metric)

    def test_metric_tiers_and_units(self):
        assert get_metric("area").unit_kind == UnitKind.AREA
        assert get_metric("perimeter").unit_kind == UnitKind.LENGTH
        assert get_metric("circularity").unit_kind == UnitKind.RATIO
        assert get_metric("max_diameter").unit_kind == UnitKind.LENGTH
        for key in ("area", "perimeter", "circularity", "max_diameter"):
            assert get_metric(key).tier == Tier.GEOMETRY
            assert get_metric(key).value_dim == 1

    def test_get_metric_unknown_key_raises(self):
        with pytest.raises(KeyError):
            get_metric("does_not_exist")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError):
            @register_metric
            class DuplicateArea(Metric):
                key = "area"  # already registered
                name = "Dup"
                description = ""
                tier = Tier.GEOMETRY
                unit_kind = UnitKind.AREA

                def compute_batch(self, ctx, params=None):
                    return {}


class TestCatalogSerialization:
    def test_catalog_is_serializable(self):
        import json

        catalog = list_metrics()
        # JSON round-trips without error -> fully serializable.
        json.dumps(catalog)
        keys = {entry["key"] for entry in catalog}
        assert {"area", "perimeter", "circularity", "max_diameter"} <= keys

    def test_catalog_entry_shape(self):
        entry = next(e for e in list_metrics() if e["key"] == "area")
        assert entry["tier"] == "geometry"
        assert entry["unit_kind"] == "area"
        assert entry["value_dim"] == 1
        assert entry["params_schema"] is None
        # Single-component metrics have no component names.
        assert entry["components"] is None
        assert set(entry) == {
            "key", "name", "description", "tier", "unit_kind", "value_dim", "components",
            "params_schema",
        }


class TestResolveUnit:
    def test_length_and_area_units(self):
        assert resolve_unit(UnitKind.LENGTH, "mm") == "mm"
        assert resolve_unit(UnitKind.AREA, "mm") == "mm²"
        assert resolve_unit(UnitKind.LENGTH, "px") == "px"
        assert resolve_unit(UnitKind.AREA, "px") == "px²"

    def test_unitless_kinds(self):
        for kind in (UnitKind.RATIO, UnitKind.COUNT, UnitKind.COLOR, UnitKind.INTENSITY, UnitKind.NONE):
            assert resolve_unit(kind, "mm") == ""


class TestMetricsMatchQuantificationModel:
    """The registry metrics must reproduce QuantificationModel.from_contour exactly."""

    @pytest.mark.parametrize("scale_x,scale_y,unit", [
        (1.0, 1.0, "px"),
        (0.5, 0.5, "mm"),
        (0.5, 0.25, "mm"),  # anisotropic
    ])
    def test_geometry_matches(self, scale_x, scale_y, unit):
        contour = make_rectangle_contour()
        ctx = QuantContext(
            contours=[contour],
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            scale_x=scale_x,
            scale_y=scale_y,
            unit=unit,
        )
        # Reference values from the legacy dual-write vehicle.
        points_px = np.stack([
            np.asarray(contour.x) * IMAGE_WIDTH,
            np.asarray(contour.y) * IMAGE_HEIGHT,
        ], axis=-1)
        ref = QuantificationModel.from_contour(points_px, scale_x=scale_x, scale_y=scale_y, unit=unit)

        assert get_metric("area").compute_batch(ctx)[1][0] == pytest.approx(ref.area)
        assert get_metric("perimeter").compute_batch(ctx)[1][0] == pytest.approx(ref.perimeter)
        assert get_metric("circularity").compute_batch(ctx)[1][0] == pytest.approx(ref.circularity)
        assert get_metric("max_diameter").compute_batch(ctx)[1][0] == pytest.approx(ref.max_diameter)


class TestQuantContext:
    def test_siblings_and_units(self):
        parent = Contour(id=1, x=[0.0, 0.5, 0.5], y=[0.0, 0.0, 0.5], parent_id=None)
        child_a = Contour(id=2, x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2], parent_id=1)
        child_b = Contour(id=3, x=[0.3, 0.4, 0.4], y=[0.3, 0.3, 0.4], parent_id=1)
        ctx = QuantContext([parent, child_a, child_b], width=100, height=100, unit="mm")

        siblings = ctx.siblings_of(child_a)
        assert {c.id for c in siblings} == {3}
        assert {c.id for c in ctx.siblings_of(child_a, include_self=True)} == {2, 3}
        assert ctx.resolve_unit(UnitKind.AREA) == "mm²"

    def test_image_loader_decodes_once(self):
        calls = {"n": 0}

        def loader():
            calls["n"] += 1
            return np.zeros((4, 4, 3), dtype=np.uint8)

        ctx = QuantContext([], width=4, height=4, image_loader=loader)
        assert ctx.image is not None
        _ = ctx.image  # cached, must not decode again
        assert calls["n"] == 1

    def test_image_none_without_loader(self):
        ctx = QuantContext([], width=4, height=4)
        assert ctx.image is None
