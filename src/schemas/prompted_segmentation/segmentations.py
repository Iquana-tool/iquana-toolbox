from logging import getLogger
from typing import List

from pydantic import BaseModel, Field

from ..contours import Contour
from .prompts import Prompts

logger = getLogger(__name__)


class PromptedSegmentationWebsocketRequest(BaseModel):
    """ Model for 2D segmentation form data. """
    model_identifier: str = Field(..., title="Model identifier", description="Model identifier string. "
                                                                             "Used to select the model.")
    user_id: str | int = Field(..., title="User ID", description="Unique identifier for the user.")
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: list[list[bool]] | None = Field(None, title="Previous Mask",
                                                  description="Optional previous mask to provide context.")


class SemanticSegmentationMask(BaseModel):
    """ Model for semantic masks. """
    contours: List[Contour] = Field(default=[], description="List of objects represented by their contours.")
    confidence: float = Field(default=0.0, description="Confidence score of the prompted_segmentation. This can be a predicted"
                                                       " IoU for example.")

    @classmethod
    def from_mask_and_score(cls, mask, score):
        """ Create a semantic segmentation mask from a mask and a score. """


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SemanticSegmentationMask]
    image_id: int = 0
    model: str
