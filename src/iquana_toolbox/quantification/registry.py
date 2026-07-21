"""Metric registry for extensible object quantification.

Every metric is a small class registered under a unique string key. The registry is
the single source of truth for which metrics exist, what they measure (tier), how
their values relate to physical units (unit kind) and how many components a value
has (e.g. 3 for a LAB color). Metrics compute in batch over a per-image
:class:`~iquana_toolbox.quantification.context.QuantContext`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from logging import getLogger
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from iquana_toolbox.quantification.context import QuantContext

logger = getLogger(__name__)


class Tier(str, Enum):
    """Coarse grouping of metrics by the information they need to compute."""
    GEOMETRY = "geometry"        # Contour points only.
    APPEARANCE = "appearance"    # Needs the image pixels.
    RELATIONAL = "relational"    # Needs sibling / parent contours.
    CONTEXTUAL = "contextual"    # Needs the whole image context.


class UnitKind(str, Enum):
    """Physical kind of a metric value; controls how the per-row unit string is resolved.

    LENGTH values scale with the pixel scale (unit "px", "mm", ...), AREA values with
    the pixel scale squared (unit "px²", "mm²", ...); all other kinds are unitless.
    """
    LENGTH = "length"
    AREA = "area"
    RATIO = "ratio"
    COUNT = "count"
    COLOR = "color"
    INTENSITY = "intensity"
    NONE = "none"


def resolve_unit(unit_kind: UnitKind, unit: str) -> str:
    """Resolve the unit string for a metric value given the image's length unit.

    :param unit_kind: The metric's :class:`UnitKind`.
    :param unit: The image's length unit, e.g. "px" or "mm".
    :returns: "mm" for lengths, "mm²" for areas, "" for unitless kinds.
    """
    if unit_kind == UnitKind.LENGTH:
        return unit
    if unit_kind == UnitKind.AREA:
        return f"{unit}²"
    return ""


class Metric(ABC):
    """Base class for all quantification metrics.

    Subclasses define the class attributes below and implement :meth:`compute_batch`.
    Register a metric with the :func:`register_metric` decorator; registration
    instantiates the class once and adds it to :data:`METRIC_REGISTRY`.
    """
    key: str
    name: str
    description: str
    tier: Tier
    unit_kind: UnitKind
    value_dim: int = 1  # Number of components per value, e.g. 3 for LAB color.
    # Human-readable names of each component, ordered to match the value array
    # (e.g. ("R", "G", "B") for RGB color). Defaults to None; when None the frontend
    # falls back to positional component indices. Length should equal ``value_dim``.
    components: tuple[str, ...] | None = None
    params_model: type[BaseModel] | None = None  # Optional pydantic model for parameters.

    @abstractmethod
    def compute_batch(self, ctx: "QuantContext", params: BaseModel | None = None) -> dict[int, np.ndarray]:
        """Compute this metric for every target contour in the context.

        :param ctx: The per-image computation context.
        :param params: Optional parameters, an instance of ``params_model`` if defined.
        :returns: Mapping from contour id to a value array of shape ``(value_dim,)``.
        """


METRIC_REGISTRY: dict[str, Metric] = {}


def register_metric(metric_cls: type[Metric]) -> type[Metric]:
    """Class decorator that instantiates a metric and registers it under its key.

    :raises ValueError: If a metric with the same key is already registered.
    """
    instance = metric_cls()
    if instance.key in METRIC_REGISTRY:
        raise ValueError(f"A metric with key '{instance.key}' is already registered "
                         f"({type(METRIC_REGISTRY[instance.key]).__name__}).")
    METRIC_REGISTRY[instance.key] = instance
    logger.debug(f"Registered metric '{instance.key}' ({metric_cls.__name__}).")
    return metric_cls


def get_metric(key: str) -> Metric:
    """Return the registered metric for ``key``.

    :raises KeyError: If no metric is registered under ``key``.
    """
    try:
        return METRIC_REGISTRY[key]
    except KeyError:
        raise KeyError(f"No metric registered under key '{key}'. "
                       f"Known metrics: {sorted(METRIC_REGISTRY)}") from None


def list_metrics() -> list[dict[str, Any]]:
    """Serializable catalog of all registered metrics.

    :returns: One dict per metric with key, name, description, tier, unit_kind,
        value_dim and the JSON schema of the params model (or None).
    """
    catalog = []
    for metric in METRIC_REGISTRY.values():
        catalog.append({
            "key": metric.key,
            "name": metric.name,
            "description": metric.description,
            "tier": metric.tier.value,
            "unit_kind": metric.unit_kind.value,
            "value_dim": metric.value_dim,
            "components": list(metric.components) if metric.components else None,
            "params_schema": metric.params_model.model_json_schema() if metric.params_model else None,
        })
    return catalog
