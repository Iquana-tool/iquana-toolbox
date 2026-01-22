from functools import cached_property

import numpy as np
from pycocotools import mask as maskUtils
from pydantic import BaseModel, Field
from PIL import Image

from typing import Union

from src.schemas.caches import get_image_from_url_cached
from src.schemas.masks import BinaryMask
from src.schemas.prompted_segmentation.prompts import Prompts


# --- Base Model ---

class BaseImageRequest(BaseModel):
    """ Shared fields and logic for all image-based requests. """
    image_url: str = Field(..., title="Image URL")
    model_registry_key: str = Field(..., title="Model registry key", description="Model identifier string.")
    user_id: Union[str, int] = Field(..., title="User ID", description="Unique identifier for the user.")

    class Config:
        # This allows property to work smoothly with Pydantic
        ignored_types = (property, cached_property)

    @property
    def image(self) -> Image.Image:
        """ Shared logic to open the image. """
        # You might want to add error handling here (e.g., requests.get for remote URLs)
        return get_image_from_url_cached(self.image_url)


# --- Concrete Implementations ---

class PromptedSegmentationRequest(BaseImageRequest):
    """ Model for prompted segmentation. """
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: BinaryMask | None = Field(None, title="Previous Mask")


class CompletionRequest(BaseImageRequest):
    """ Model for instance discovery with image exemplars and concepts. """
    exemplars: list[BinaryMask] = Field(..., description="Exemplars is a list of RLE encoded binary masks")
    negative_exemplars: list[BinaryMask] | None = Field(..., title="Negative exemplars")
    concept: str | None = Field(default=None, description="Optional string describing the concept.")

    @cached_property
    def positive_exemplar_masks(self) -> list[np.ndarray]:
        return [exemplar.mask for exemplar in self.exemplars]

    @cached_property
    def negative_exemplar_masks(self) -> list[np.ndarray]:
        return [exemplar.mask for exemplar in self.negative_exemplars]

    @cached_property
    def combined_exemplar_mask(self) -> np.ndarray:
        combined_mask = self.exemplars[0].mask
        if len(self.exemplars) > 1:
            for exemplar in self.exemplars[1:]:
                combined_mask = np.logical_or(combined_mask, exemplar.mask)
        return combined_mask
