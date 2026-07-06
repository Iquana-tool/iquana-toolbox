"""Tests for the appearance-tier metrics (mean_color_rgb, mean_color_lab, mean_intensity).

Uses synthetic solid-color images so expected values are known exactly (or to a tight
tolerance for LAB, which is a nonlinear conversion). Also verifies the ``ctx.image``
decode-once contract and the grayscale / empty-mask handling documented in
``metrics/appearance.py``.
"""
import cv2 as cv
import numpy as np
import pytest

from iquana_toolbox.quantification import METRIC_REGISTRY, QuantContext, Tier, UnitKind, get_metric
from iquana_toolbox.schemas.database.contours import Contour

IMAGE_SIZE = 100


def make_square_contour(contour_id=1, x0=20, y0=20, x1=80, y1=80, size=IMAGE_SIZE):
    """A square contour in pixel box [x0, x1) x [y0, y1), stored normalized."""
    x_px = [x0, x1, x1, x0]
    y_px = [y0, y0, y1, y1]
    return Contour(
        id=contour_id,
        x=[x / size for x in x_px],
        y=[y / size for y in y_px],
    )


def red_image(size=IMAGE_SIZE) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # pure red, RGB order
    return img


class TestRegistration:
    def test_appearance_metrics_registered(self):
        for key in ("mean_color_rgb", "mean_color_lab", "mean_intensity"):
            assert key in METRIC_REGISTRY
            assert get_metric(key).tier == Tier.APPEARANCE

    def test_value_dims_and_unit_kinds(self):
        assert get_metric("mean_color_rgb").value_dim == 3
        assert get_metric("mean_color_lab").value_dim == 3
        assert get_metric("mean_intensity").value_dim == 1
        assert get_metric("mean_color_rgb").unit_kind == UnitKind.COLOR
        assert get_metric("mean_color_lab").unit_kind == UnitKind.COLOR
        assert get_metric("mean_intensity").unit_kind == UnitKind.INTENSITY


class TestMeanColorOnSolidRed:
    def test_mean_color_rgb_is_pure_red(self):
        contour = make_square_contour()
        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE,
                           image_loader=lambda: red_image())
        values = get_metric("mean_color_rgb").compute_batch(ctx)
        assert values[1] == pytest.approx([255.0, 0.0, 0.0])

    def test_mean_intensity_matches_cv_gray_weighting(self):
        contour = make_square_contour()
        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE,
                           image_loader=lambda: red_image())
        values = get_metric("mean_intensity").compute_batch(ctx)
        expected_gray = cv.cvtColor(red_image(), cv.COLOR_RGB2GRAY)[20:80, 20:80].mean()
        assert values[1][0] == pytest.approx(expected_gray)

    def test_mean_color_lab_is_consistent_with_opencv_conversion(self):
        contour = make_square_contour()
        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE,
                           image_loader=lambda: red_image())
        values = get_metric("mean_color_lab").compute_batch(ctx)
        expected_lab = cv.cvtColor(red_image(), cv.COLOR_RGB2LAB)[20:80, 20:80].reshape(-1, 3).mean(axis=0)
        assert values[1] == pytest.approx(expected_lab, abs=1e-6)
        # Pure red in opencv's 8-bit LAB scaling is a known-ish point: high L, high a (red-green
        # axis skewed toward red), b roughly mid-to-high (red-yellow axis).
        L, a, b = values[1]
        assert 100 < L < 200
        assert a > 150  # strongly toward the "red" end of the a* axis


class TestGrayscaleImages:
    def test_grayscale_2d_image_yields_equal_rgb_channels(self):
        contour = make_square_contour()
        gray_value = 123

        def loader():
            return np.full((IMAGE_SIZE, IMAGE_SIZE), gray_value, dtype=np.uint8)

        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE, image_loader=loader)
        values = get_metric("mean_color_rgb").compute_batch(ctx)
        assert values[1] == pytest.approx([gray_value, gray_value, gray_value])

    def test_grayscale_hw1_image_yields_equal_rgb_channels(self):
        contour = make_square_contour()
        gray_value = 200

        def loader():
            return np.full((IMAGE_SIZE, IMAGE_SIZE, 1), gray_value, dtype=np.uint8)

        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE, image_loader=loader)
        values = get_metric("mean_color_rgb").compute_batch(ctx)
        assert values[1] == pytest.approx([gray_value, gray_value, gray_value])
        intensity = get_metric("mean_intensity").compute_batch(ctx)
        assert intensity[1][0] == pytest.approx(gray_value)


class TestEmptyMask:
    def test_contour_outside_image_yields_zeros(self):
        """A contour with no points rasterizes to an all-False mask -> zeros, not NaN/crash."""
        contour = Contour(id=1, x=[], y=[])
        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE,
                           image_loader=lambda: red_image())
        assert get_metric("mean_color_rgb").compute_batch(ctx)[1] == pytest.approx([0.0, 0.0, 0.0])
        assert get_metric("mean_color_lab").compute_batch(ctx)[1] == pytest.approx([0.0, 0.0, 0.0])
        assert get_metric("mean_intensity").compute_batch(ctx)[1] == pytest.approx([0.0])

    def test_no_image_loader_yields_zeros(self):
        """Without an image_loader, ctx.image is None; appearance metrics degrade to zeros
        rather than raising, so calling them accidentally on a geometry-only context is safe."""
        contour = make_square_contour()
        ctx = QuantContext([contour], width=IMAGE_SIZE, height=IMAGE_SIZE)
        assert get_metric("mean_color_rgb").compute_batch(ctx)[1] == pytest.approx([0.0, 0.0, 0.0])
        assert get_metric("mean_intensity").compute_batch(ctx)[1] == pytest.approx([0.0])


class TestImageDecodedOnce:
    def test_multiple_metrics_over_multiple_contours_decode_image_once(self):
        calls = {"n": 0}

        def loader():
            calls["n"] += 1
            return red_image()

        contours = [
            make_square_contour(contour_id=1, x0=0, y0=0, x1=50, y1=50),
            make_square_contour(contour_id=2, x0=50, y0=50, x1=100, y1=100),
        ]
        ctx = QuantContext(contours, width=IMAGE_SIZE, height=IMAGE_SIZE, image_loader=loader)

        get_metric("mean_color_rgb").compute_batch(ctx)
        get_metric("mean_color_lab").compute_batch(ctx)
        get_metric("mean_intensity").compute_batch(ctx)

        assert calls["n"] == 1  # cached_property on QuantContext.image ensures a single decode
