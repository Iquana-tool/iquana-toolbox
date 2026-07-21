"""Pure geometry functions shared by the quantification metrics and the legacy
:class:`~iquana_toolbox.schemas.database.quantification.QuantificationModel`.

This module must stay free of schema imports so it can be used from both sides
without circular imports. All functions accept points as ``(N, 2)`` arrays or the
opencv contour shape ``(N, 1, 2)`` and are robust against degenerate contours.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np
from scipy.spatial.distance import pdist


def as_point_array(points) -> np.ndarray:
    """Coerce points to a ``(N, 2)`` float64 array.

    :param points: Points of shape ``(N, 2)`` or the opencv contour shape ``(N, 1, 2)``.
    :returns: A float64 array of shape ``(N, 2)``.
    """
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def area_and_perimeter(points) -> tuple[float, float]:
    """Enclosed area and closed perimeter of a contour polygon.

    Degenerate contours (fewer than 3 points) cannot enclose an area and return ``(0.0, 0.0)``.
    """
    points = as_point_array(points)
    if points.shape[0] < 3:
        return 0.0, 0.0
    # Opencv contours have the form Number of points x empty dimension x (x, y).
    cv_contour = points.astype(np.float32).reshape(-1, 1, 2)
    return float(cv.contourArea(cv_contour)), float(cv.arcLength(cv_contour, True))


def circularity(area: float, perimeter: float) -> float:
    """Dimensionless circularity ``4 * pi * area / perimeter**2`` (1.0 for a perfect circle).

    Returns 0.0 for degenerate inputs (zero area or perimeter).
    """
    if area == 0 or perimeter == 0:
        return 0.0
    return float((4 * np.pi * area) / (perimeter ** 2))


def max_diameter(points) -> float:
    """Maximum pairwise distance between any two contour points (0.0 for < 2 points)."""
    points = as_point_array(points)
    if points.shape[0] < 2:
        return 0.0
    distances = pdist(points, "euclidean")
    return float(np.max(distances)) if distances.size > 0 else 0.0


def polygon_centroid(points) -> np.ndarray:
    """Centroid of the contour polygon (falls back to the vertex mean for degenerate shapes).

    :returns: ``[cx, cy]`` array; ``[nan, nan]`` when the contour has no points.
    """
    points = as_point_array(points)
    if points.shape[0] == 0:
        return np.array([np.nan, np.nan])
    cv_contour = points.astype(np.float32).reshape(-1, 1, 2)
    moments = cv.moments(cv_contour)
    if moments["m00"] != 0:
        return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return points.mean(axis=0)
