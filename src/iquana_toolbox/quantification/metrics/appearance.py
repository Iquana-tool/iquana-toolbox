"""Appearance-tier metrics: mean color (RGB / CIELAB) and mean intensity.

Unlike geometry metrics, these need the decoded image pixels (``ctx.image``), which is
why they live in a separate tier and are computed lazily (see
``app.services.quantification.compute_appearance_metrics_for_dataset`` in the backend).

Efficiency: :meth:`Metric.compute_batch` is called once per image with every target
contour for that image. Each metric here decodes/converts ``ctx.image`` to the color
space it needs exactly ONCE per call (not per contour) and then indexes it with each
contour's boolean mask.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np
from pydantic import BaseModel

from iquana_toolbox.quantification.context import QuantContext
from iquana_toolbox.quantification.registry import Metric, Tier, UnitKind, register_metric


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Promote a possibly-grayscale image to 3-channel ``uint8`` RGB.

    Handles ``(H, W)`` and ``(H, W, 1)`` grayscale arrays (e.g. from ``color_mode='L'``
    images) by replicating the single channel, so grayscale images yield R=G=B and every
    downstream color/LAB conversion still sees a 3-channel image. Non-``uint8`` arrays are
    clipped and cast (appearance metrics operate in the 0-255 range).
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _mask_for_contour(ctx: QuantContext, contour, rgb: np.ndarray) -> np.ndarray:
    """Boolean mask for ``contour`` rescaled to ``rgb``'s actual H x W.

    ``to_binary_mask`` takes explicit height/width and rasterizes the contour's
    normalized coordinates into that grid, so passing the image's own shape (rather than
    ``ctx.width`` / ``ctx.height``) guards against any mismatch between the stored image
    dimensions and the decoded array. A contour with fewer than 3 points cannot enclose
    any pixels (and would crash opencv's ``fillPoly``), so it short-circuits to an
    all-``False`` mask — the empty-mask case handled uniformly by the metrics below.
    """
    height, width = rgb.shape[0], rgb.shape[1]
    if len(contour.x) < 3:
        return np.zeros((height, width), dtype=bool)
    return contour.to_binary_mask(height=height, width=width)


@register_metric
class MeanColorRgbMetric(Metric):
    """Mean R, G, B of the object's pixels (0-255 each)."""
    key = "mean_color_rgb"
    name = "Mean color (RGB)"
    description = ("Mean red, green and blue channel value (0-255 each) over the "
                   "contour's filled pixels. Empty masks (contour outside the image, "
                   "or zero-area) yield [0, 0, 0].")
    tier = Tier.APPEARANCE
    unit_kind = UnitKind.COLOR
    value_dim = 3  # components: 0=R, 1=G, 2=B
    components = ("R", "G", "B")

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        if ctx.image is None:
            return {contour.id: np.zeros(3, dtype=np.float64) for contour in ctx.contours}
        rgb = _as_rgb_uint8(ctx.image)  # decoded/promoted once for all contours
        for contour in ctx.contours:
            mask = _mask_for_contour(ctx, contour, rgb)
            pixels = rgb[mask]
            if pixels.size == 0:
                result[contour.id] = np.zeros(3, dtype=np.float64)
            else:
                result[contour.id] = pixels.mean(axis=0).astype(np.float64)
        return result


@register_metric
class MeanColorLabMetric(Metric):
    """Mean CIELAB color of the object's pixels — the primary appearance metric.

    Averaging in CIELAB is perceptually meaningful (unlike averaging RGB directly,
    which can produce muddy, non-representative colors for multi-hued regions). Uses
    opencv's ``cv.COLOR_RGB2LAB`` conversion, which for ``uint8`` input scales all three
    channels into the 0-255 range (L: 0-255 maps to L*: 0-100, a*/b*: -127-127 mapped to
    0-255), NOT the conventional L*: 0-100 / a*,b*: -128..127 floating point range.
    """
    key = "mean_color_lab"
    name = "Mean color (CIELAB)"
    description = ("Mean CIELAB color (opencv 8-bit scaling: L,a,b each in 0-255, "
                   "not the conventional L*:0-100/a*,b*:-128..127 range) over the "
                   "contour's filled pixels, averaged in LAB space for perceptual "
                   "meaningfulness. Empty masks yield [0, 0, 0].")
    tier = Tier.APPEARANCE
    unit_kind = UnitKind.COLOR
    value_dim = 3  # components: 0=L, 1=a, 2=b (opencv 8-bit scaling)
    components = ("L", "a", "b")

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        if ctx.image is None:
            return {contour.id: np.zeros(3, dtype=np.float64) for contour in ctx.contours}
        rgb = _as_rgb_uint8(ctx.image)
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)  # converted once for all contours
        for contour in ctx.contours:
            mask = _mask_for_contour(ctx, contour, lab)
            pixels = lab[mask]
            if pixels.size == 0:
                result[contour.id] = np.zeros(3, dtype=np.float64)
            else:
                result[contour.id] = pixels.mean(axis=0).astype(np.float64)
        return result


@register_metric
class MeanIntensityMetric(Metric):
    """Mean grayscale luminance of the object's pixels (0-255)."""
    key = "mean_intensity"
    name = "Mean intensity"
    description = ("Mean grayscale luminance (0-255), computed via opencv's standard "
                   "RGB to grayscale weighting, over the contour's filled pixels. "
                   "Empty masks yield 0.")
    tier = Tier.APPEARANCE
    unit_kind = UnitKind.INTENSITY
    value_dim = 1

    def compute_batch(self, ctx: QuantContext, params: BaseModel | None = None) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        if ctx.image is None:
            return {contour.id: np.zeros(1, dtype=np.float64) for contour in ctx.contours}
        rgb = _as_rgb_uint8(ctx.image)
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)  # converted once for all contours
        for contour in ctx.contours:
            mask = _mask_for_contour(ctx, contour, gray[:, :, np.newaxis])
            pixels = gray[mask]
            if pixels.size == 0:
                result[contour.id] = np.zeros(1, dtype=np.float64)
            else:
                result[contour.id] = np.array([pixels.mean()], dtype=np.float64)
        return result
