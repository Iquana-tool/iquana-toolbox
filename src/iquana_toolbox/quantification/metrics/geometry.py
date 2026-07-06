"""Geometry-tier metrics: area, perimeter, circularity, max_diameter.

These are the four legacy metrics, ported to the registry. They compute in physical
units (pixels scaled by the image's per-axis scale) and share their math with the
legacy :class:`~iquana_toolbox.schemas.database.quantification.QuantificationModel`
through :mod:`iquana_toolbox.quantification.geometry_math` — no formula is duplicated.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from iquana_toolbox.quantification import geometry_math as gm
from iquana_toolbox.quantification.context import QuantContext
from iquana_toolbox.quantification.registry import Metric, Tier, UnitKind, register_metric


@register_metric
class AreaMetric(Metric):
    """Enclosed area of the contour, in the image length unit squared."""
    key = "area"
    name = "Area"
    description = "Enclosed area of the contour polygon."
    tier = Tier.GEOMETRY
    unit_kind = UnitKind.AREA

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for contour in ctx.contours:
            area, _ = gm.area_and_perimeter(ctx.points_physical(contour))
            result[contour.id] = np.array([area], dtype=np.float64)
        return result


@register_metric
class PerimeterMetric(Metric):
    """Closed perimeter (arc length) of the contour, in the image length unit."""
    key = "perimeter"
    name = "Perimeter"
    description = "Closed perimeter (arc length) of the contour polygon."
    tier = Tier.GEOMETRY
    unit_kind = UnitKind.LENGTH

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for contour in ctx.contours:
            _, perimeter = gm.area_and_perimeter(ctx.points_physical(contour))
            result[contour.id] = np.array([perimeter], dtype=np.float64)
        return result


@register_metric
class CircularityMetric(Metric):
    """Dimensionless circularity 4*pi*area/perimeter**2 (1.0 for a perfect circle)."""
    key = "circularity"
    name = "Circularity"
    description = "Dimensionless shape descriptor 4*pi*area/perimeter^2 (1.0 for a circle)."
    tier = Tier.GEOMETRY
    unit_kind = UnitKind.RATIO

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for contour in ctx.contours:
            points = ctx.points_physical(contour)
            area, perimeter = gm.area_and_perimeter(points)
            result[contour.id] = np.array([gm.circularity(area, perimeter)], dtype=np.float64)
        return result


@register_metric
class MaxDiameterMetric(Metric):
    """Maximum pairwise distance between any two contour points, in the image length unit."""
    key = "max_diameter"
    name = "Max Diameter"
    description = "Maximum distance between any two points of the contour."
    tier = Tier.GEOMETRY
    unit_kind = UnitKind.LENGTH

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        for contour in ctx.contours:
            diameter = gm.max_diameter(ctx.points_physical(contour))
            result[contour.id] = np.array([diameter], dtype=np.float64)
        return result
