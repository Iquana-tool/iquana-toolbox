"""Helpers for turning raw model predictions into annotations that fit an existing mask.

Batch inference produces contours that nothing has reconciled yet: several models may
propose the same object, and a child-level prediction does not know which parent instance
it belongs to. :mod:`iquana_toolbox.inference.overlap` answers both questions with the same
primitive -- rasterized mask overlap.
"""
from iquana_toolbox.inference.overlap import (
    DEFAULT_RASTER_SIZE,
    NmsResult,
    Suppression,
    best_parent,
    boxes_overlap,
    containment,
    mask_iou,
    nms,
    rasterize,
)

__all__ = [
    "DEFAULT_RASTER_SIZE",
    "NmsResult",
    "Suppression",
    "best_parent",
    "boxes_overlap",
    "containment",
    "mask_iou",
    "nms",
    "rasterize",
]
