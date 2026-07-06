"""Extensible object quantification: a metric registry, a per-image context and the
built-in geometry metrics.

Importing this package registers the built-in metrics (via
:mod:`iquana_toolbox.quantification.metrics`), so ``METRIC_REGISTRY`` is populated as
soon as the package is imported.
"""
from iquana_toolbox.quantification.context import QuantContext
from iquana_toolbox.quantification.registry import (
    METRIC_REGISTRY,
    Metric,
    Tier,
    UnitKind,
    get_metric,
    list_metrics,
    register_metric,
    resolve_unit,
)

# Import the metrics package for its registration side effects.
from iquana_toolbox.quantification import metrics  # noqa: E402,F401

__all__ = [
    "METRIC_REGISTRY",
    "Metric",
    "Tier",
    "UnitKind",
    "QuantContext",
    "get_metric",
    "list_metrics",
    "register_metric",
    "resolve_unit",
    "metrics",
]
