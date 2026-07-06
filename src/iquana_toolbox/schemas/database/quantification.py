from logging import getLogger

import numpy as np
from pydantic import BaseModel, Field

from iquana_toolbox.quantification import geometry_math as gm

logger = getLogger(__name__)


class QuantificationModel(BaseModel):
    """ A model to keep track of quantifications and enable easy extensions to the quantifications."""
    area: float | None = Field(default=None, description="Area of the contour. Expressed in `unit` squared.")
    perimeter: float | None = Field(default=None, description="Perimeter of the contour. Expressed in `unit`.")
    circularity: float | None = Field(default=None, description="Circularity of the contour. Dimensionless.")
    max_diameter: float | None = Field(default=None, description="Maximum distance of any two points in "
                                                                 "the contour. Expressed in `unit`.")
    unit: str | None = Field(default="px", description="Unit the metric values are expressed in. Lengths "
                                                       "(perimeter, max_diameter) are in this unit, area is in "
                                                       "this unit squared.")

    @property
    def is_empty(self) -> bool:
        return self.area is None

    @classmethod
    def from_contour(cls,
                     points_px: np.ndarray,
                     scale_x: float = 1.0,
                     scale_y: float = 1.0,
                     unit: str = "px") -> "QuantificationModel":
        """
        Compute quantification metrics from a contour given in PIXEL coordinates.

        The points are converted to physical units by multiplying x-coordinates with ``scale_x`` and
        y-coordinates with ``scale_y`` BEFORE any metric is computed. This makes all metrics exact even
        for anisotropic pixels (scale_x != scale_y): area scales by scale_x * scale_y, lengths follow the
        anisotropically scaled geometry, and circularity is computed in physical space.

        :param points_px: Contour points in pixel space. Accepts shape (N, 2) or the opencv
            contour shape (N, 1, 2).
        :param scale_x: Physical size of one pixel along x (e.g. mm per pixel). Defaults to 1.
        :param scale_y: Physical size of one pixel along y (e.g. mm per pixel). Defaults to 1.
        :param unit: Unit that scale_x / scale_y are expressed in. Defaults to "px" (raw pixels).
        :returns: A QuantificationModel with area, perimeter, circularity and max_diameter populated.
        """
        points = gm.as_point_array(points_px)
        # Convert to physical units before computing anything.
        points = points * np.array([scale_x, scale_y], dtype=np.float64)

        # All geometry is computed by the shared geometry_math functions so the metric
        # registry and this legacy dual-write vehicle can never drift apart. The helpers
        # already handle degenerate contours (area/perimeter -> 0, circularity -> 0).
        area, perimeter = gm.area_and_perimeter(points)
        if points.shape[0] < 3:
            logger.debug(f"Degenerate contour with {points.shape[0]} point(s); metrics set to zero.")
        return cls(
            area=area,
            perimeter=perimeter,
            circularity=gm.circularity(area, perimeter),
            max_diameter=gm.max_diameter(points),
            unit=unit,
        )
