"""Relational-tier metrics: quantities derived from a contour's position in the
parent/child hierarchy of its image (as opposed to the CONTEXTUAL tier, which measures
spatial relationships between same-parent siblings).

The single metric here today is ``n_children``: how many contours in the image name this
contour as their parent. Like the contextual tier this is RELATIONAL - a contour's value
depends on OTHER contours (its children) - but the invalidation is PARENT-TARGETED, not
sibling-group-wide: ``n_children`` only changes when a CHILD is added, removed or
re-parented, which affects exactly the (old/new) parent, never the parent's siblings. See
``app.services.quantification.compute_relational_metrics_for_dataset`` and
``mark_relational_stale_for_parent`` in the backend for how that staleness is wired.

Unlike the contextual metrics, ``n_children`` is defined and meaningful for EVERY contour,
including leaves (0 children is a real value, not a missing one), so it returns a value for
every target contour in the context.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from iquana_toolbox.quantification.context import QuantContext
from iquana_toolbox.quantification.registry import Metric, Tier, UnitKind, register_metric


@register_metric
class NumberOfChildrenMetric(Metric):
    """Number of contours in the image whose ``parent_id`` equals this contour's ``id``."""
    key = "n_children"
    name = "Number of children"
    description = ("Number of contours in the same image that name this contour as their "
                   "parent (i.e. how many direct child objects it contains). Defined for "
                   "every contour: a leaf object with no children has value 0 (a real, "
                   "meaningful count, not a missing value), so unlike the contextual "
                   "nearest-neighbour metrics no contour is omitted from the result.")
    tier = Tier.RELATIONAL
    unit_kind = UnitKind.COUNT

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        # ctx holds ALL of the image's contours (children share the parent's image/mask), so
        # a single pass over ctx.contours counts, per parent_id, how many contours point to
        # it. Contours whose parent_id is None (root level) contribute to no parent's count.
        children_per_parent: dict[int, int] = {}
        for contour in ctx.contours:
            if contour.parent_id is not None:
                children_per_parent[contour.parent_id] = children_per_parent.get(contour.parent_id, 0) + 1

        result: dict[int, np.ndarray] = {}
        for contour in ctx.contours:
            count = children_per_parent.get(contour.id, 0)
            result[contour.id] = np.array([count], dtype=np.float64)
        return result
