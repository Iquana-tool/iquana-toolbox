"""Tests for the contextual-tier metrics (nn_distance, mean_knn_distance).

Uses contours placed at known pixel centroids so the expected nearest-neighbour
distances are known exactly (in both pixel and physical space, when scale != 1).
"""
import numpy as np
import pytest

from iquana_toolbox.quantification import METRIC_REGISTRY, QuantContext, Tier, UnitKind, get_metric
from iquana_toolbox.schemas.database.contours import Contour

IMAGE_SIZE = 1000


def make_point_contour(contour_id, cx_px, cy_px, parent_id=None, half=5, size=IMAGE_SIZE):
    """A small square contour centered at (cx_px, cy_px), stored normalized.

    ``half`` is half the side length in pixels; small relative to the gaps used between
    contours in these tests so the centroid is (to floating point precision) at
    (cx_px, cy_px).
    """
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return Contour(
        id=contour_id,
        parent_id=parent_id,
        x=[x / size for x in x_px],
        y=[y / size for y in y_px],
    )


class TestRegistration:
    def test_contextual_metrics_registered(self):
        for key in ("nn_distance", "mean_knn_distance"):
            assert key in METRIC_REGISTRY
            metric = get_metric(key)
            assert metric.tier == Tier.CONTEXTUAL
            assert metric.unit_kind == UnitKind.LENGTH
            assert metric.value_dim == 1


class TestNnDistanceSymmetricPair:
    def test_two_siblings_are_each_others_nearest_neighbor(self):
        # Two contours 100px apart under the same parent.
        c1 = make_point_contour(1, 100, 100, parent_id=10)
        c2 = make_point_contour(2, 200, 100, parent_id=10)
        ctx = QuantContext([c1, c2], width=IMAGE_SIZE, height=IMAGE_SIZE)
        values = get_metric("nn_distance").compute_batch(ctx)
        assert set(values) == {1, 2}
        assert values[1][0] == pytest.approx(100.0)
        assert values[2][0] == pytest.approx(100.0)

    def test_physical_scale_is_applied(self):
        # Same geometry as above but with a physical scale != 1 (2 mm / px, isotropic).
        c1 = make_point_contour(1, 100, 100, parent_id=10)
        c2 = make_point_contour(2, 200, 100, parent_id=10)
        ctx = QuantContext([c1, c2], width=IMAGE_SIZE, height=IMAGE_SIZE,
                           scale_x=2.0, scale_y=2.0, unit="mm")
        values = get_metric("nn_distance").compute_batch(ctx)
        assert values[1][0] == pytest.approx(200.0)  # 100px * 2mm/px
        assert values[2][0] == pytest.approx(200.0)
        assert ctx.resolve_unit(get_metric("nn_distance").unit_kind) == "mm"


class TestOnlyChildOmitted:
    def test_lone_contour_is_omitted_not_nan_or_zero(self):
        c1 = make_point_contour(1, 500, 500, parent_id=10)
        ctx = QuantContext([c1], width=IMAGE_SIZE, height=IMAGE_SIZE)
        values = get_metric("nn_distance").compute_batch(ctx)
        assert values == {}

        knn_values = get_metric("mean_knn_distance").compute_batch(ctx)
        assert knn_values == {}


class TestThreeInALine:
    """Three siblings in a line: A --100px-- B --300px-- C. B's nearest neighbor is A
    (distance 100), not C (distance 300)."""

    def _contours(self):
        # Offset well away from the (0, 0) coordinate-clamping edge (Contour.x/y are
        # clamped to [0, 1], which would distort centroids near pixel 0).
        a = make_point_contour(1, 100, 100, parent_id=10)
        b = make_point_contour(2, 200, 100, parent_id=10)
        c = make_point_contour(3, 500, 100, parent_id=10)
        return [a, b, c]

    def test_middle_contours_nearest_neighbor_is_the_closer_one(self):
        ctx = QuantContext(self._contours(), width=IMAGE_SIZE, height=IMAGE_SIZE)
        values = get_metric("nn_distance").compute_batch(ctx)
        assert values[1][0] == pytest.approx(100.0)  # A -> B
        assert values[2][0] == pytest.approx(100.0)  # B -> A (closer than B -> C = 300)
        assert values[3][0] == pytest.approx(300.0)  # C -> B

    def test_mean_knn_distance_with_k_larger_than_group_uses_all_others(self):
        # MEAN_KNN_K defaults to 3, but each contour here only has 2 possible neighbors,
        # so k should clamp to (group_size - 1) = 2 and average both other distances.
        ctx = QuantContext(self._contours(), width=IMAGE_SIZE, height=IMAGE_SIZE)
        values = get_metric("mean_knn_distance").compute_batch(ctx)
        # A: distances to B (100) and C (400) -> mean 250.
        assert values[1][0] == pytest.approx((100.0 + 400.0) / 2)
        # B: distances to A (100) and C (300) -> mean 200.
        assert values[2][0] == pytest.approx((100.0 + 300.0) / 2)
        # C: distances to A (400) and B (300) -> mean 350.
        assert values[3][0] == pytest.approx((400.0 + 300.0) / 2)


class TestRootLevelContours:
    """Contours with parent_id=None are siblings of every other root contour in the
    same image (documented behavior)."""

    def test_root_level_contours_get_nearest_neighbor_among_other_roots(self):
        c1 = make_point_contour(1, 100, 100, parent_id=None)
        c2 = make_point_contour(2, 150, 100, parent_id=None)
        c3 = make_point_contour(3, 600, 100, parent_id=None)
        ctx = QuantContext([c1, c2, c3], width=IMAGE_SIZE, height=IMAGE_SIZE)
        values = get_metric("nn_distance").compute_batch(ctx)
        assert values[1][0] == pytest.approx(50.0)
        assert values[2][0] == pytest.approx(50.0)
        assert values[3][0] == pytest.approx(450.0)

    def test_root_and_nested_groups_do_not_mix(self):
        # A root-level pair and a separate parent group; nn_distance must not leak
        # across groups even though physically closer.
        size = 2000
        root_a = make_point_contour(1, 100, 100, parent_id=None, size=size)
        root_b = make_point_contour(2, 1100, 100, parent_id=None, size=size)  # far from root_a
        child_a = make_point_contour(3, 110, 110, parent_id=99, size=size)  # very close to root_a
        child_b = make_point_contour(4, 120, 110, parent_id=99, size=size)
        ctx = QuantContext([root_a, root_b, child_a, child_b], width=size, height=size)
        values = get_metric("nn_distance").compute_batch(ctx)
        # root_a's only sibling is root_b (far away) - the nearby child contours (a
        # different parent group) must NOT be considered.
        assert values[1][0] == pytest.approx(1000.0)
        assert values[2][0] == pytest.approx(1000.0)
        # children only see each other.
        assert values[3][0] == pytest.approx(10.0)
        assert values[4][0] == pytest.approx(10.0)


class TestSingleKdTreePerGroup:
    """Structural check that one KDTree is built per parent group, not per contour
    (i.e. compute_batch does not degrade to O(n^2) pairwise distance calls)."""

    def test_kdtree_constructed_once_per_group(self, monkeypatch):
        import iquana_toolbox.quantification.metrics.contextual as contextual_mod

        calls = {"n": 0}
        original_cls = contextual_mod.cKDTree

        class _CountingKDTree(original_cls):
            def __init__(self, *args, **kwargs):
                calls["n"] += 1
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(contextual_mod, "cKDTree", _CountingKDTree)

        # Two separate parent groups of 4 contours each -> exactly 2 KDTree builds.
        group1 = [make_point_contour(i, 100 + i * 20, 100, parent_id=1) for i in range(4)]
        group2 = [make_point_contour(i + 10, 100 + i * 20, 500, parent_id=2) for i in range(4)]
        ctx = QuantContext(group1 + group2, width=IMAGE_SIZE, height=IMAGE_SIZE)

        get_metric("nn_distance").compute_batch(ctx)
        assert calls["n"] == 2
