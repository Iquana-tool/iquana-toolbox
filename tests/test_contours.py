"""Tests for contour normalization and conversion in iquana_toolbox."""
import numpy as np
import pytest

from iquana_toolbox.schemas.database.contours import (
    Contour,
    normalize_contours,
)
from iquana_toolbox.schemas.database.masks import BinaryMask


def test_normalize_contours_preserves_fractions_for_int32():
    """An int32 OpenCV contour on a non-square image must normalize to fractional float coordinates."""
    # Image size: width=800, height=600 (non-square)
    width = 800
    height = 600

    # OpenCV contours are int32 with shape (N, 1, 2)
    # Coordinates include zero, interior points, and edges
    raw_contour = np.array([
        [[0, 0]],
        [[100, 150]],
        [[400, 300]],
        [[800, 600]],
    ], dtype=np.int32)

    raw_copy = raw_contour.copy()
    normalized = normalize_contours([raw_contour], width=width, height=height)[0]

    # Input array must not be mutated
    np.testing.assert_array_equal(raw_contour, raw_copy)

    # Output must be floating point
    assert np.issubdtype(normalized.dtype, np.floating)

    # Check exact fractional values
    expected = np.array([
        [[0.0, 0.0]],
        [[100.0 / 800.0, 150.0 / 600.0]],  # 0.125, 0.25
        [[400.0 / 800.0, 300.0 / 600.0]],  # 0.5, 0.5
        [[1.0, 1.0]],
    ], dtype=np.float64)
    np.testing.assert_allclose(normalized, expected, atol=1e-7)


def test_from_binary_mask_multi_component_nondegenerate():
    """Contour.from_binary_mask with only_return_biggest_contour=False must preserve all components as non-zero."""
    # Create non-square mask with two disconnected square components
    height, width = 400, 600
    mask = np.zeros((height, width), dtype=bool)

    # Component 1: top-left box [50:100, 50:100]
    mask[50:100, 50:100] = True
    # Component 2: bottom-right box [250:350, 400:500] (bigger)
    mask[250:350, 400:500] = True

    contours = Contour.from_binary_mask(
        mask,
        only_return_biggest_contour=False,
        label_id=42,
        added_by="test_model",
        confidence=0.95,
    )

    assert len(contours) == 2

    for c in contours:
        assert isinstance(c, Contour)
        assert c.label_id == 42
        assert c.added_by == "test_model"
        assert c.confidence == 0.95
        assert len(c.x) >= 3
        assert len(c.y) >= 3

        # Coordinates must be strictly in [0, 1] and NOT truncated to 0
        x_arr = np.array(c.x)
        y_arr = np.array(c.y)
        assert np.any(x_arr > 0.0), "X coordinates were truncated to zero!"
        assert np.any(y_arr > 0.0), "Y coordinates were truncated to zero!"
        assert np.all((x_arr >= 0.0) & (x_arr <= 1.0))
        assert np.all((y_arr >= 0.0) & (y_arr <= 1.0))

    # Verify relative positioning of the two components
    # Component 1 center is ~ (75/600, 75/400) = (0.125, 0.1875)
    # Component 2 center is ~ (450/600, 300/400) = (0.75, 0.75)
    means_x = [np.mean(c.x) for c in contours]
    assert min(means_x) < 0.25
    assert max(means_x) > 0.65


def test_from_binary_mask_biggest_contour_and_empty():
    """Test only_return_biggest_contour=True and empty mask behavior."""
    height, width = 200, 300
    mask = np.zeros((height, width), dtype=bool)

    # Empty mask returns empty list
    empty_result = Contour.from_binary_mask(mask, only_return_biggest_contour=True)
    assert empty_result == []

    # Single component
    mask[40:80, 50:110] = True
    contour = Contour.from_binary_mask(mask, only_return_biggest_contour=True, label_id=1)
    assert isinstance(contour, Contour)
    assert contour.label_id == 1
    assert len(contour.x) >= 3
    assert np.any(np.array(contour.x) > 0.0)


def test_from_binary_mask_model_and_fit_to_mask():
    """Test from_binary_mask_model and fit_to_mask conversions."""
    height, width = 300, 400
    mask = np.zeros((height, width), dtype=bool)
    mask[50:150, 60:160] = True

    mask_model = BinaryMask.from_numpy_array(mask, score=0.88)
    contour = Contour.from_binary_mask_model(mask_model, label_id=5)
    assert isinstance(contour, Contour)
    assert contour.confidence == 0.88
    assert contour.label_id == 5
    assert np.any(np.array(contour.x) > 0.0)

    # Test round trip to binary mask
    reconstructed_mask = contour.to_binary_mask(height=height, width=width)
    assert reconstructed_mask.shape == (height, width)
    # Intersection should be high
    intersection = np.logical_and(mask, reconstructed_mask).sum()
    union = np.logical_or(mask, reconstructed_mask).sum()
    assert (intersection / union) > 0.95

    # Test fit_to_mask
    allowed = np.zeros((height, width), dtype=bool)
    allowed[50:100, 60:160] = True  # Keep only top half
    fitted_contour, changed = contour.fit_to_mask(allowed)
    assert changed is True
    assert isinstance(fitted_contour, Contour)
    assert np.max(fitted_contour.y) <= (101.0 / height)
