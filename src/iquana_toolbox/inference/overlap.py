"""Mask-overlap primitives: deduplicate predictions, and find a prediction's parent.

Two problems in batch inference reduce to "how much do these two contours overlap":

* **Duplicates.** Several models may be pointed at the same label, and a patching run adds
  predictions on top of annotations that already exist. Greedy NMS keeps the
  highest-scoring proposal of each overlapping cluster and drops the rest.
* **Hierarchy.** A child-level prediction (a nucleus) arrives with no ``parent_id``. The
  parent is the already-written contour that *contains* it best -- containment, not IoU,
  because a nucleus is much smaller than its cell and their IoU is therefore always low.

Coordinates and the raster grid
-------------------------------
:class:`~iquana_toolbox.schemas.database.contours.Contour` stores coordinates normalized to
``[0, 1]``, so every contour is rasterized onto the same *square* grid regardless of the
image's aspect ratio. That is not an approximation: mapping normalized space to a square is
an affine scaling, and an affine map multiplies every area by the same constant -- so the
intersection/union and intersection/area *ratios* this module computes are exactly the ones
you would get on the real image. Only the sampling resolution is approximate, and
:data:`DEFAULT_RASTER_SIZE` is chosen so that a contour covering 0.4% of the image still
lands on ~1000 pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from iquana_toolbox.schemas.database.contours import Contour

#: Side length of the square grid contours are rasterized onto. Overlap of two contours is
#: measured on this grid, so it trades accuracy for speed: 512x512 booleans are 256 KiB
#: each, and a whole image's worth of instances rasterizes in milliseconds.
DEFAULT_RASTER_SIZE = 512


def rasterize(contour: Contour, size: int = DEFAULT_RASTER_SIZE) -> np.ndarray:
    """Fill ``contour`` onto a ``(size, size)`` boolean grid.

    Args:
        contour: The contour to rasterize (coordinates normalized to ``[0, 1]``).
        size: Side length of the square grid.

    Returns:
        A ``(size, size)`` boolean array, True inside the contour.
    """
    return contour.to_binary_mask(height=size, width=size)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over union of two boolean masks. ``0.0`` when both are empty."""
    intersection = int(np.count_nonzero(first & second))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of ``inner`` that lies inside ``outer`` (``|inner ∩ outer| / |inner|``).

    Asymmetric on purpose. This is the right measure for parent lookup: a small object fully
    inside a large one scores 1.0, where their IoU would be near zero.
    """
    area = int(np.count_nonzero(inner))
    if area == 0:
        return 0.0
    return int(np.count_nonzero(inner & outer)) / area


def boxes_overlap(first: Contour, second: Contour) -> bool:
    """Whether two contours' axis-aligned bounding boxes intersect.

    A cheap reject used before rasterizing: instances on an image are mostly disjoint, so
    this turns the quadratic overlap scan into a nearly linear one.
    """
    a_min_x, a_min_y, a_max_x, a_max_y = first.get_bbox()
    b_min_x, b_min_y, b_max_x, b_max_y = second.get_bbox()
    return not (a_max_x < b_min_x or b_max_x < a_min_x
                or a_max_y < b_min_y or b_max_y < a_min_y)


class _RasterCache:
    """Rasterize each contour at most once per NMS pass, keyed by identity."""

    def __init__(self, size: int):
        self._size = size
        self._masks: dict[int, np.ndarray] = {}

    def get(self, contour: Contour) -> np.ndarray:
        mask = self._masks.get(id(contour))
        if mask is None:
            mask = rasterize(contour, self._size)
            self._masks[id(contour)] = mask
        return mask


@dataclass(frozen=True)
class Suppression:
    """One dropped candidate and what beat it."""

    index: int
    """Position of the dropped candidate in the input list."""
    iou: float
    """Overlap with the winner that suppressed it."""
    against_existing: bool
    """True when the winner was a pre-existing annotation rather than another candidate."""


@dataclass
class NmsResult:
    """Outcome of one :func:`nms` pass."""

    kept: list[int] = field(default_factory=list)
    """Indices of the surviving candidates, in descending score order."""
    suppressed: list[Suppression] = field(default_factory=list)
    """Dropped candidates, with the overlap that dropped them."""

    @property
    def suppressed_indices(self) -> set[int]:
        return {item.index for item in self.suppressed}


def nms(
    candidates: Sequence[Contour],
    *,
    scores: Sequence[float] | None = None,
    iou_threshold: float = 0.7,
    existing: Iterable[Contour] = (),
    raster_size: int = DEFAULT_RASTER_SIZE,
) -> NmsResult:
    """Greedy non-maximum suppression over predicted contours.

    Candidates are visited in descending score order; each one is dropped when it overlaps
    an already-accepted candidate -- or *any* contour in ``existing`` -- by more than
    ``iou_threshold``.

    ``existing`` is the asymmetry that makes this usable for patching: those contours are
    already in the database (drawn by a human, or written by an earlier level of the same
    job) and are never removed. They only ever suppress incoming predictions, so a patching
    run can add to a mask without ever destroying work that is already there.

    Args:
        candidates: The predicted contours to filter.
        scores: One score per candidate; defaults to each candidate's ``confidence``.
        iou_threshold: Overlap above which the lower-scoring contour is dropped.
        existing: Contours that suppress but are never suppressed.
        raster_size: Side length of the square grid overlap is measured on.

    Returns:
        An :class:`NmsResult` naming the survivors and what each drop lost to.
    """
    if not candidates:
        return NmsResult()
    if scores is None:
        scores = [contour.confidence for contour in candidates]
    if len(scores) != len(candidates):
        raise ValueError("scores must have one entry per candidate.")

    cache = _RasterCache(raster_size)
    existing = list(existing)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

    result = NmsResult()
    accepted: list[Contour] = []
    for index in order:
        candidate = candidates[index]
        overlap = _first_overlap(candidate, existing, iou_threshold, cache)
        if overlap is not None:
            result.suppressed.append(Suppression(index, overlap, against_existing=True))
            continue
        overlap = _first_overlap(candidate, accepted, iou_threshold, cache)
        if overlap is not None:
            result.suppressed.append(Suppression(index, overlap, against_existing=False))
            continue
        result.kept.append(index)
        accepted.append(candidate)
    return result


def _first_overlap(
    candidate: Contour,
    others: Sequence[Contour],
    iou_threshold: float,
    cache: _RasterCache,
) -> float | None:
    """The IoU of the first contour in ``others`` that overlaps ``candidate`` too much."""
    for other in others:
        if not boxes_overlap(candidate, other):
            continue
        iou = mask_iou(cache.get(candidate), cache.get(other))
        if iou > iou_threshold:
            return iou
    return None


def best_parent(
    child: Contour,
    parents: Sequence[Contour],
    *,
    min_containment: float = 0.5,
    raster_size: int = DEFAULT_RASTER_SIZE,
) -> int | None:
    """Index of the parent that contains ``child`` best, or ``None`` if none qualifies.

    Ties are broken by area (the *smallest* qualifying parent wins), so a nucleus inside a
    cell inside a colony attaches to the cell rather than the colony when both are offered
    as candidates.

    Args:
        child: The contour to place.
        parents: Candidate parents, already restricted to the parent label.
        min_containment: Minimum fraction of the child that must lie inside a parent.
        raster_size: Side length of the square grid overlap is measured on.

    Returns:
        The index into ``parents`` of the winning parent, or ``None`` when the child lies
        outside every candidate (the caller decides whether to drop it or keep it at root).
    """
    if not parents:
        return None
    cache = _RasterCache(raster_size)
    child_mask = cache.get(child)

    best_index: int | None = None
    best_key: tuple[float, float] | None = None
    for index, parent in enumerate(parents):
        if not boxes_overlap(child, parent):
            continue
        parent_mask = cache.get(parent)
        score = containment(child_mask, parent_mask)
        if score < min_containment:
            continue
        # Prefer the highest containment; among equally containing parents, the tightest fit.
        key = (score, -float(np.count_nonzero(parent_mask)))
        if best_key is None or key > best_key:
            best_key, best_index = key, index
    return best_index
