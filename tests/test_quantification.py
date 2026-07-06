""" Tests for pixel-space quantification of contours.

Regression context: quantification used to be computed from NORMALIZED [0, 1] contour
coordinates, which anisotropically distorted shapes on non-square images (wrong
circularity / max_diameter) and produced area / perimeter in meaningless
"fraction of image" units. All metrics must be computed from PIXEL coordinates,
optionally scaled to physical units.
"""
import numpy as np
import pytest

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.quantification import QuantificationModel

# Non-square image: normalized coordinates would distort shapes here.
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 500


def make_rectangle_contour():
    """A 100 x 50 px rectangle on a 1000 x 500 image, stored normalized."""
    x_px = [100, 200, 200, 100]
    y_px = [100, 100, 150, 150]
    return Contour(
        x=[x / IMAGE_WIDTH for x in x_px],
        y=[y / IMAGE_HEIGHT for y in y_px],
    )


def make_circle_contour(radius_px=100, center=(150, 150), n_points=360):
    """A circle (radius in px) on a 1000 x 500 image, stored normalized."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x_px = center[0] + radius_px * np.cos(angles)
    y_px = center[1] + radius_px * np.sin(angles)
    return Contour(
        x=(x_px / IMAGE_WIDTH).tolist(),
        y=(y_px / IMAGE_HEIGHT).tolist(),
    )


class TestContourConstruction:
    def test_contour_constructible_without_image_dimensions(self):
        """A Contour must not compute quantification as a validator side effect."""
        contour = make_rectangle_contour()
        assert contour.quantification is None

    def test_explicit_quantification_is_kept(self):
        quant = QuantificationModel(area=1.0, perimeter=2.0, circularity=3.0, max_diameter=4.0)
        contour = Contour(x=[0.1, 0.2, 0.2], y=[0.1, 0.1, 0.2], quantification=quant)
        assert contour.quantification.area == 1.0


class TestPixelSpaceMetrics:
    def test_rectangle_area_and_perimeter_in_pixels(self):
        contour = make_rectangle_contour()
        quant = contour.compute_quantification(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        assert quant is contour.quantification
        assert quant.area == pytest.approx(5000.0)  # 100 px * 50 px
        assert quant.perimeter == pytest.approx(300.0)  # 2 * (100 + 50) px
        assert quant.max_diameter == pytest.approx(np.hypot(100, 50))
        assert quant.unit == "px"

    def test_circle_circularity_on_non_square_image(self):
        """Regression: normalized coords on a 1000x500 image turn a circle into a
        2:1 ellipse, tanking circularity. Pixel-space metrics must not."""
        contour = make_circle_contour(radius_px=100)
        quant = contour.compute_quantification(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        assert quant.circularity == pytest.approx(1.0, abs=0.01)
        assert quant.max_diameter == pytest.approx(200.0, rel=0.01)
        assert quant.area == pytest.approx(np.pi * 100 ** 2, rel=0.01)

    def test_old_normalized_computation_would_fail_circularity(self):
        """Sanity check documenting the old bug: metrics from normalized coords are wrong."""
        contour = make_circle_contour(radius_px=100)
        wrong = QuantificationModel.from_contour(contour.points)  # normalized coords, NOT pixels
        assert wrong.circularity < 0.95  # anisotropic distortion breaks circularity


class TestPhysicalScaling:
    def test_scale_converts_units(self):
        contour = make_rectangle_contour()
        quant = contour.compute_quantification(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            scale_x=0.5, scale_y=0.5, unit="mm",
        )
        assert quant.area == pytest.approx(5000.0 * 0.5 * 0.5)  # mm^2
        assert quant.perimeter == pytest.approx(300.0 * 0.5)  # mm
        assert quant.max_diameter == pytest.approx(np.hypot(100, 50) * 0.5)  # mm
        assert quant.unit == "mm"

    def test_isotropic_scale_keeps_circularity_dimensionless(self):
        contour = make_circle_contour(radius_px=100)
        quant = contour.compute_quantification(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            scale_x=0.25, scale_y=0.25, unit="um",
        )
        assert quant.circularity == pytest.approx(1.0, abs=0.01)


class TestDegenerateContours:
    def test_empty_contour(self):
        contour = Contour(x=[], y=[])
        quant = contour.compute_quantification(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        assert quant.area == 0.0
        assert quant.perimeter == 0.0
        assert quant.circularity == 0.0
        assert quant.max_diameter == 0.0

    def test_single_point_contour(self):
        quant = QuantificationModel.from_contour(np.array([[10.0, 10.0]]))
        assert quant.area == 0.0
        assert quant.max_diameter == 0.0

    def test_two_point_contour(self):
        quant = QuantificationModel.from_contour(np.array([[0.0, 0.0], [3.0, 4.0]]))
        assert quant.area == 0.0
        assert quant.max_diameter == pytest.approx(5.0)

    def test_zero_area_collinear_contour(self):
        quant = QuantificationModel.from_contour(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        assert quant.area == 0.0
        assert quant.circularity == 0.0
        assert quant.max_diameter == pytest.approx(np.hypot(2, 2))


class TestInputShapes:
    def test_accepts_opencv_contour_shape(self):
        """QuantificationModel.from_contour accepts the opencv (N, 1, 2) shape."""
        points = np.array([[100, 100], [200, 100], [200, 150], [100, 150]], dtype=np.float32)
        cv_shaped = points.reshape(-1, 1, 2)
        quant = QuantificationModel.from_contour(cv_shaped)
        assert quant.area == pytest.approx(5000.0)
        assert quant.perimeter == pytest.approx(300.0)
