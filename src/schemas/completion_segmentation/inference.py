from typing import List
from pydantic import BaseModel, Field


class CompletionMainAPIRequest(BaseModel):
    image_id: int | None = Field(...)
    seed_contour_ids: List[int] = Field(...)
    model_key: str = Field(...)


class CompletionServiceRequest(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: list[dict] = Field(..., description="Seeds is a list of rle encoded binary masks")
    concept: str | None = Field(default=None, description="Optional str describing the concept of the objects to be detected.")
