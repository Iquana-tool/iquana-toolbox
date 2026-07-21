"""Per-image computation context for quantification metrics.

A :class:`QuantContext` bundles everything a metric needs for one image: the target
contours (with normalized coordinates, parent links and label ids), the image
geometry (size + physical scale) and an optional lazy image loader for appearance
metrics. Metrics read from the context in :meth:`Metric.compute_batch`.

Kept as plain python (not pydantic) so numpy arrays and cached properties are cheap.
"""
from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Callable

import numpy as np

from iquana_toolbox.quantification.geometry_math import polygon_centroid
from iquana_toolbox.quantification.registry import UnitKind, resolve_unit

if TYPE_CHECKING:
    from iquana_toolbox.schemas.database.contours import Contour


class QuantContext:
    """Everything a metric needs to compute over a single image's contours.

    :param contours: Target contours for this image. Each must expose ``id``,
        normalized ``x`` / ``y`` lists, ``parent_id`` and ``label_id``.
    :param width: Image width in pixels.
    :param height: Image height in pixels.
    :param scale_x: Physical size of one pixel along x (e.g. mm per pixel).
    :param scale_y: Physical size of one pixel along y (e.g. mm per pixel).
    :param unit: Length unit the scales are expressed in ("px", "mm", ...).
    :param image_loader: Optional callable returning the image as an RGB ``(H, W, 3)``
        numpy array. Used by appearance metrics (Step 3+); decoded at most once
        via :attr:`image`.
    """

    def __init__(
            self,
            contours: list["Contour"],
            width: int,
            height: int,
            scale_x: float = 1.0,
            scale_y: float = 1.0,
            unit: str = "px",
            image_loader: Callable[[], np.ndarray] | None = None,
    ):
        self.contours = contours
        self.width = int(width)
        self.height = int(height)
        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)
        self.unit = unit
        self._image_loader = image_loader

    @cached_property
    def image(self) -> np.ndarray | None:
        """The image as an RGB array, decoded at most once. ``None`` if no loader was given."""
        if self._image_loader is None:
            return None
        return self._image_loader()

    @cached_property
    def _contours_by_parent(self) -> dict[int | None, list["Contour"]]:
        """Index of contours grouped by ``parent_id`` (None = image / root level)."""
        index: dict[int | None, list["Contour"]] = {}
        for contour in self.contours:
            index.setdefault(contour.parent_id, []).append(contour)
        return index

    def _scale_vector(self) -> np.ndarray:
        return np.array([self.scale_x, self.scale_y], dtype=np.float64)

    def points_px(self, contour: "Contour") -> np.ndarray:
        """Contour points as an ``(N, 2)`` float array in raw pixel coordinates."""
        if not contour.x:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack([
            np.asarray(contour.x, dtype=np.float64) * self.width,
            np.asarray(contour.y, dtype=np.float64) * self.height,
        ], axis=-1)

    def points_physical(self, contour: "Contour") -> np.ndarray:
        """Contour points in physical units (pixels scaled by ``scale_x`` / ``scale_y``)."""
        return self.points_px(contour) * self._scale_vector()

    def centroid_px(self, contour: "Contour") -> np.ndarray:
        """Polygon centroid of the contour in raw pixel coordinates ``[cx, cy]``."""
        return polygon_centroid(self.points_px(contour))

    def centroid_physical(self, contour: "Contour") -> np.ndarray:
        """Polygon centroid of the contour in physical units (``centroid_px`` scaled by
        ``scale_x`` / ``scale_y``). Shared here so contextual metrics (Step 4+) that need
        physical-space distances between centroids don't duplicate the scaling logic."""
        return self.centroid_px(contour) * self._scale_vector()

    def siblings_of(self, contour: "Contour", include_self: bool = False) -> list["Contour"]:
        """Contours sharing the same ``parent_id`` (None = image level).

        :param include_self: Whether to include ``contour`` itself in the result.
        """
        siblings = self._contours_by_parent.get(contour.parent_id, [])
        if include_self:
            return list(siblings)
        return [c for c in siblings if c.id != contour.id or c is not contour]

    def resolve_unit(self, unit_kind: UnitKind) -> str:
        """Per-row unit string for a metric of the given unit kind (see :func:`resolve_unit`)."""
        return resolve_unit(unit_kind, self.unit)
