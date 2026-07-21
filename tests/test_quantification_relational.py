"""Tests for the RELATIONAL-tier ``n_children`` metric.

Verifies the metric is registered under the RELATIONAL tier as a value_dim-1 COUNT metric,
counts children correctly at the QuantContext level (parent with k children -> k; leaf -> 0),
and reflects a re-parent (moving a child to a different parent shifts the counts).
"""
import numpy as np
import pytest

from iquana_toolbox.quantification import (
    METRIC_REGISTRY,
    QuantContext,
    Tier,
    UnitKind,
    get_metric,
    list_metrics,
)
from iquana_toolbox.schemas.database.contours import Contour


def _tri(x0=0.0, y0=0.0):
    """A tiny triangle contour offset by (x0, y0), normalized to [0, 1]."""
    return ([x0, x0 + 0.05, x0 + 0.05], [y0, y0, y0 + 0.05])


def _make_context(contours):
    return QuantContext(contours, width=100, height=100, unit="px")


class TestRelationalRegistration:
    def test_registered_under_relational_tier(self):
        assert "n_children" in METRIC_REGISTRY
        metric = get_metric("n_children")
        assert metric.tier == Tier.RELATIONAL
        assert metric.unit_kind == UnitKind.COUNT
        assert metric.value_dim == 1

    def test_catalog_entry(self):
        entry = next(e for e in list_metrics() if e["key"] == "n_children")
        assert entry["tier"] == "relational"
        assert entry["unit_kind"] == "count"
        assert entry["value_dim"] == 1
        assert entry["name"] == "Number of children"

    def test_count_resolves_to_empty_unit(self):
        ctx = _make_context([])
        assert ctx.resolve_unit(get_metric("n_children").unit_kind) == ""


class TestNumberOfChildren:
    def test_parent_with_children_and_leaves(self):
        # parent(1) has 3 children (2,3,4); child 2 itself has 1 child (5).
        px, py = _tri()
        parent = Contour(id=1, x=px, y=py, parent_id=None)
        c2 = Contour(id=2, x=px, y=py, parent_id=1)
        c3 = Contour(id=3, x=px, y=py, parent_id=1)
        c4 = Contour(id=4, x=px, y=py, parent_id=1)
        c5 = Contour(id=5, x=px, y=py, parent_id=2)
        ctx = _make_context([parent, c2, c3, c4, c5])

        result = get_metric("n_children").compute_batch(ctx)
        # Every contour gets a value, including leaves (0 is meaningful).
        assert set(result) == {1, 2, 3, 4, 5}
        assert result[1][0] == pytest.approx(3.0)  # parent -> 3 children
        assert result[2][0] == pytest.approx(1.0)  # c2 -> 1 child (c5)
        assert result[3][0] == pytest.approx(0.0)  # leaf
        assert result[4][0] == pytest.approx(0.0)  # leaf
        assert result[5][0] == pytest.approx(0.0)  # leaf

    def test_leaf_only_context(self):
        px, py = _tri()
        leaves = [Contour(id=i, x=px, y=py, parent_id=None) for i in (10, 11, 12)]
        ctx = _make_context(leaves)
        result = get_metric("n_children").compute_batch(ctx)
        assert all(result[i][0] == pytest.approx(0.0) for i in (10, 11, 12))

    def test_reparent_shifts_counts(self):
        px, py = _tri()
        # Start: parent A(1) has child C(3); parent B(2) has no children.
        parent_a = Contour(id=1, x=px, y=py, parent_id=None)
        parent_b = Contour(id=2, x=px, y=py, parent_id=None)
        child = Contour(id=3, x=px, y=py, parent_id=1)

        before = get_metric("n_children").compute_batch(_make_context([parent_a, parent_b, child]))
        assert before[1][0] == pytest.approx(1.0)  # A has the child
        assert before[2][0] == pytest.approx(0.0)  # B has none

        # Re-parent the child from A to B.
        child.parent_id = 2
        after = get_metric("n_children").compute_batch(_make_context([parent_a, parent_b, child]))
        assert after[1][0] == pytest.approx(0.0)  # A lost the child
        assert after[2][0] == pytest.approx(1.0)  # B gained it
        assert after[3][0] == pytest.approx(0.0)  # child is still a leaf

    def test_value_shape_is_scalar(self):
        px, py = _tri()
        ctx = _make_context([Contour(id=1, x=px, y=py, parent_id=None)])
        value = get_metric("n_children").compute_batch(ctx)[1]
        assert isinstance(value, np.ndarray)
        assert value.shape == (1,)
