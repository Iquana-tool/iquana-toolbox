"""Contextual-tier metrics: nearest-neighbour distance and mean distance to the k
nearest neighbours, both scoped to same-parent siblings (root-level contours, i.e.
``parent_id is None``, are treated as siblings of every other root contour in the same
image, since a root object does have a meaningful nearest neighbour among other roots).

Unlike geometry/appearance metrics (computed per contour, independent of the rest of the
image), these are RELATIONAL: a contour's value depends on where its siblings are. See
``app.services.quantification.compute_contextual_metrics_for_dataset`` in the backend for
how staleness is expanded to the whole parent group before recomputing.

Efficiency: contours are grouped by ``parent_id`` and ONE ``scipy.spatial.cKDTree`` is
built per group from the siblings' physical-space centroids, then queried once for the
whole group (O(n log n) per group) rather than computing an O(n^2) pairwise distance
matrix.

Distances are computed in PHYSICAL space (``ctx.centroid_physical``), matching how the
existing geometry-tier lengths (perimeter, max_diameter) are already reported in the
image's physical unit (mm, ...) rather than raw pixels.

Only-child groups (no siblings) have no meaningful nearest-neighbour value: the metrics
below OMIT such contours from the returned dict entirely (rather than writing 0 or NaN),
so ``compute_and_store_metrics`` never persists a corrupting 0 for a lone object and the
per-metric aggregation (mean/std/...) naturally excludes it. See the module docstring of
``app.services.quantification`` for the delete-then-insert correctness this depends on.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from scipy.spatial import cKDTree

from iquana_toolbox.quantification.context import QuantContext
from iquana_toolbox.quantification.registry import Metric, Tier, UnitKind, register_metric

if TYPE_CHECKING:
    from iquana_toolbox.schemas.database.contours import Contour

# k for mean_knn_distance. Fixed for now (Step 4): params_model / param-threading through
# compute_and_store_metrics is not wired up yet (see registry.Metric.params_model
# docstring and the Step 4 task notes) - a future step can turn this into a
# params_model-controlled value. Kept as a module constant so it's easy to find.
MEAN_KNN_K = 3


def _group_by_parent(ctx: QuantContext) -> dict[int | None, list["Contour"]]:
    """Group ``ctx.contours`` by ``parent_id`` (None = root/image level).

    Root-level contours (``parent_id is None``) are grouped together so they are
    considered siblings of one another, exactly like any other parent group.
    """
    groups: dict[int | None, list["Contour"]] = {}
    for contour in ctx.contours:
        groups.setdefault(contour.parent_id, []).append(contour)
    return groups


def _centroids(ctx: QuantContext, contours: list["Contour"]) -> np.ndarray:
    """Physical-space centroids for a list of contours, shape ``(N, 2)``."""
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([ctx.centroid_physical(c) for c in contours], axis=0)


@register_metric
class NearestNeighborDistanceMetric(Metric):
    """Euclidean distance from a contour's centroid to its nearest same-parent sibling's
    centroid, in physical units."""
    key = "nn_distance"
    name = "Nearest-neighbour distance"
    description = ("Euclidean distance (in the image's physical length unit) from this "
                   "contour's centroid to the centroid of the nearest OTHER contour "
                   "sharing the same parent (root-level contours are siblings of every "
                   "other root contour in the image). Only-child contours (no siblings) "
                   "have no meaningful value and are OMITTED from the result.")
    tier = Tier.CONTEXTUAL
    unit_kind = UnitKind.LENGTH

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for _parent_id, siblings in _group_by_parent(ctx).items():
            if len(siblings) < 2:
                continue  # only-child: no meaningful neighbor, omit entirely.
            centroids = _centroids(ctx, siblings)
            tree = cKDTree(centroids)
            # k=2: the nearest neighbor of each point is itself (distance 0), so the
            # second column is the nearest OTHER point.
            distances, _indices = tree.query(centroids, k=2)
            for contour, dist in zip(siblings, distances[:, 1]):
                result[contour.id] = np.array([dist], dtype=np.float64)
        return result


@register_metric
class MeanKnnDistanceMetric(Metric):
    """Mean distance to up to :data:`MEAN_KNN_K` nearest same-parent siblings, in
    physical units."""
    key = "mean_knn_distance"
    name = f"Mean distance to {MEAN_KNN_K} nearest neighbours"
    description = (f"Mean euclidean distance (in the image's physical length unit) from "
                   f"this contour's centroid to its up-to-{MEAN_KNN_K} nearest same-parent "
                   f"sibling centroids (fewer if the parent group is smaller; root-level "
                   f"contours are siblings of every other root contour in the image). "
                   f"Only-child contours (no siblings) have no meaningful value and are "
                   f"OMITTED from the result. k is fixed at {MEAN_KNN_K} for now.")
    tier = Tier.CONTEXTUAL
    unit_kind = UnitKind.LENGTH

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for _parent_id, siblings in _group_by_parent(ctx).items():
            n = len(siblings)
            if n < 2:
                continue  # only-child: no meaningful neighbor, omit entirely.
            k = min(MEAN_KNN_K, n - 1)
            centroids = _centroids(ctx, siblings)
            tree = cKDTree(centroids)
            # k+1: column 0 is each point itself (distance 0), columns 1..k are the k
            # nearest OTHER siblings. k+1 >= 2 always, so distances is always 2-D here
            # (scipy only squeezes to 1-D when k=1 is passed to query(), never k=2+).
            distances, _indices = tree.query(centroids, k=k + 1)
            neighbor_distances = distances[:, 1:k + 1]
            means = neighbor_distances.mean(axis=1)
            for contour, mean_dist in zip(siblings, means):
                result[contour.id] = np.array([mean_dist], dtype=np.float64)
        return result
