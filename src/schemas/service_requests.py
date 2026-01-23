import warnings
from functools import cached_property
from typing import Union, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .caches import get_image_from_url_cached
from .labels import Label, LabelHierarchy
from .masks import BinaryMask
from .prompted_segmentation.prompts import Prompts


# --- Base Model ---

class BaseImageRequest(BaseModel):
    """ Shared fields and logic for all image-based requests. """
    image_url: str = Field(..., title="Image URL")
    user_id: Union[str, int] = Field(..., title="User ID", description="Unique identifier for the user.")

    class Config:
        # This allows property to work smoothly with Pydantic
        ignored_types = (property, cached_property)

    @property
    def image(self) -> Image.Image:
        """ Shared logic to open the image. """
        # You might want to add error handling here (e.g., requests.get for remote URLs)
        return get_image_from_url_cached(self.image_url)


class BaseServiceRequest(BaseImageRequest):
    """ Expands the BaseImageRequest with a model_registry_key. """
    model_registry_key: str = Field(..., title="Model registry key", description="Model identifier string.")


# --- Concrete Implementations ---


class SemanticSegmentationRequest(BaseServiceRequest):
    """ Expands the BaseServiceRequest with a label hierarchy for the model."""
    label_hierarchy: LabelHierarchy = Field(
        ..., title="Label hierarchy",
        description="A hierarchy of the labels. Describes which labels should be present in the mask."
    )


class PromptedSegmentationRequest(BaseServiceRequest):
    """ Model for prompted segmentation. """
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: BinaryMask | None = Field(None, title="Previous Mask")


class CompletionRequest(BaseServiceRequest):
    """ Model for instance discovery with image exemplars and concepts. """
    positive_exemplars: list[BinaryMask] = Field(..., description="Exemplars is a list of RLE encoded binary masks")
    negative_exemplars: list[BinaryMask] | None = Field(..., title="Negative exemplars")
    concept: Label | None = Field(default=None, description="Optional label defining the concept.")

    @cached_property
    def positive_exemplar_masks(self) -> list[np.ndarray]:
        return [exemplar.mask for exemplar in self.positive_exemplars]

    @cached_property
    def negative_exemplar_masks(self) -> list[np.ndarray]:
        return [exemplar.mask for exemplar in self.negative_exemplars]

    @cached_property
    def combined_exemplar_mask(self) -> np.ndarray:
        combined_mask = self.positive_exemplars[0].mask
        if len(self.positive_exemplars) > 1:
            for exemplar in self.positive_exemplars[1:]:
                combined_mask = np.logical_or(combined_mask, exemplar.mask)
        return combined_mask

    def get_bboxes(self,
                   format: Literal["xywh", "x1y1x2y2", "cxcywh"] = "x1y1x2y2",
                   relative_coordinates: bool = True,
                   resize_to: None | tuple[int, int] = None) \
            -> list[list[float]]:
        bboxes = []
        for mask in self.positive_exemplars:
            if resize_to is not None and relative_coordinates:
                warnings.warn("Wanting relative coordinates and resizing to a fixed size is contradictory. "
                              "Returning resized coordinates.")
            x_min, y_min, x_max, y_max = mask.get_as_bbox(
                relative_coords=relative_coordinates if resize_to is None else True
            )

            if resize_to:
                x_min *= resize_to[1]
                y_min *= resize_to[0]
                x_max *= resize_to[1]
                y_max *= resize_to[0]

            if format == "xywh":
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            elif format == "x1y1x2y2":
                bbox = [x_min, y_min, x_max, y_max]
            elif format == "cxcywh":
                w = x_max - x_min
                h = y_max - y_min
                cx = x_min + w / 2
                cy = y_min + h / 2
                bbox = [cx, cy, w, h]
            else:
                raise ValueError("Unsupported format: {}".format(format))
            bboxes.append(bbox)
        return bboxes
