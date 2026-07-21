"""Shared, in-process vision backbones for IQUANA AI models.

These are *libraries*, not services: a model wrapper imports a backbone and owns
an instance of it in the same process. Sharing the code here (rather than running
a separate backbone service) keeps features on the same device as the task head,
avoids serializing multi-megabyte feature tensors over the network, and lets
training backprop into the head without a process boundary.
"""

from iquana_toolbox.ai.backbones.dinov3 import DINOv3Backbone

__all__ = ["DINOv3Backbone"]
