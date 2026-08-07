"""Overlap primitives used to reconcile batch-inference predictions with existing work."""
import pytest

from iquana_toolbox.inference import best_parent, containment, mask_iou, nms, rasterize
from iquana_toolbox.schemas.database.contours import Contour


def box(x0: float, y0: float, x1: float, y1: float, confidence: float = 1.0) -> Contour:
    """An axis-aligned rectangular contour in normalized coordinates."""
    return Contour(
        x=[x0, x1, x1, x0],
        y=[y0, y0, y1, y1],
        confidence=confidence,
        added_by="test",
    )


def test_identical_boxes_have_iou_one():
    first, second = box(0.1, 0.1, 0.4, 0.4), box(0.1, 0.1, 0.4, 0.4)
    assert mask_iou(rasterize(first), rasterize(second)) == pytest.approx(1.0, abs=0.01)


def test_disjoint_boxes_have_iou_zero():
    assert mask_iou(rasterize(box(0.0, 0.0, 0.2, 0.2)), rasterize(box(0.6, 0.6, 0.9, 0.9))) == 0.0


def test_iou_is_unaffected_by_the_square_raster():
    """Half-overlapping boxes score 1/3 whatever grid they are sampled on."""
    first, second = box(0.0, 0.2, 0.4, 0.8), box(0.2, 0.2, 0.6, 0.8)
    coarse = mask_iou(rasterize(first, 128), rasterize(second, 128))
    fine = mask_iou(rasterize(first, 512), rasterize(second, 512))
    assert coarse == pytest.approx(1 / 3, abs=0.02)
    assert fine == pytest.approx(1 / 3, abs=0.02)


def test_containment_is_asymmetric():
    small, large = box(0.4, 0.4, 0.5, 0.5), box(0.2, 0.2, 0.8, 0.8)
    assert containment(rasterize(small), rasterize(large)) == pytest.approx(1.0, abs=0.01)
    assert containment(rasterize(large), rasterize(small)) < 0.1


def test_nms_keeps_the_highest_scoring_of_a_duplicate_pair():
    weak, strong = box(0.1, 0.1, 0.5, 0.5, confidence=0.4), box(0.1, 0.1, 0.5, 0.5, confidence=0.9)
    result = nms([weak, strong], iou_threshold=0.7)
    assert result.kept == [1]
    assert result.suppressed_indices == {0}
    assert result.suppressed[0].against_existing is False


def test_nms_keeps_distinct_instances():
    result = nms([box(0.0, 0.0, 0.2, 0.2), box(0.6, 0.6, 0.9, 0.9)], iou_threshold=0.7)
    assert sorted(result.kept) == [0, 1]


def test_existing_contours_suppress_but_are_never_dropped():
    """A patching run must not be able to remove an annotation that is already there."""
    human = box(0.1, 0.1, 0.5, 0.5, confidence=1.0)
    prediction = box(0.1, 0.1, 0.5, 0.5, confidence=0.99)
    result = nms([prediction], existing=[human], iou_threshold=0.7)
    assert result.kept == []
    assert result.suppressed[0].against_existing is True


def test_best_parent_picks_the_tightest_container():
    child = box(0.45, 0.45, 0.55, 0.55)
    colony, cell = box(0.1, 0.1, 0.9, 0.9), box(0.4, 0.4, 0.6, 0.6)
    assert best_parent(child, [colony, cell]) == 1


def test_best_parent_returns_none_for_an_orphan():
    assert best_parent(box(0.8, 0.8, 0.9, 0.9), [box(0.1, 0.1, 0.3, 0.3)]) is None


def test_best_parent_respects_min_containment():
    """A prediction straddling a parent's edge is not silently adopted."""
    # A quarter of the child (0.4-0.5 in both axes) falls inside the parent.
    child, parent = box(0.4, 0.4, 0.6, 0.6), box(0.2, 0.2, 0.5, 0.5)
    assert best_parent(child, [parent], min_containment=0.5) is None
    assert best_parent(child, [parent], min_containment=0.2) == 0
