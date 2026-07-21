"""Built-in quantification metrics.

Importing this package imports every metric module so that the ``@register_metric``
decorators run and populate ``METRIC_REGISTRY``.
"""
from iquana_toolbox.quantification.metrics import geometry  # noqa: F401  (registration side effect)
from iquana_toolbox.quantification.metrics import appearance  # noqa: F401  (registration side effect)
from iquana_toolbox.quantification.metrics import contextual  # noqa: F401  (registration side effect)

__all__ = ["geometry", "appearance", "contextual"]
