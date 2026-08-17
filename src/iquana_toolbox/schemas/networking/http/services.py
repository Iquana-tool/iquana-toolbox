import warnings
from functools import cached_property
from typing import Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from iquana_toolbox.caches import get_image_from_url_cached
from iquana_toolbox.schemas.database.labels import Label, LabelHierarchy
from iquana_toolbox.schemas.database.masks import BinaryMask
from iquana_toolbox.schemas.prompts import Prompts

# The following contains inference requests
# --- Base Model ---

class BaseImageRequest(BaseModel):
    """ Shared fields and logic for all image-based requests. """
    image_url: str = Field(..., title="Image URL")
    user_id: Union[str, int] = Field(..., title="User ID", description="Unique identifier for the user.")

    class Config:
        # This allows property to work smoothly with Pydantic
        ignored_types = (property, cached_property)

    @property
    def image(self) -> np.ndarray:
        """ Shared logic to open the image. """
        # You might want to add error handling here (e.g., requests.get for remote URLs)
        return get_image_from_url_cached(self.image_url)


class BaseServiceRequest(BaseImageRequest):
    """ Expands the BaseImageRequest with a model_registry_key. """
    model_registry_key: str = Field(..., title="Model registry key", description="Model identifier string.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        title="Parameters",
        description="Normalized model hyperparameters and overrides.",
    )
    contour_ids: list[int] = Field(
        default_factory=list,
        title="Contour IDs",
        description="Resolved contour IDs for instance-level conditioning.",
    )
    embeddings: dict[str, list[float]] = Field(
        default_factory=dict,
        title="Embeddings",
        description="Resolved embedding vectors by kind.",
    )


# --- Concrete Implementations ---


class SemanticSegmentationRequest(BaseServiceRequest):
    """ A Semantic Segmentation Inference Request. Deprecated"""
    pass


class InstanceSegmentationRequest(BaseServiceRequest):
    """ A Instance Segmentation Inference Request."""
    label: Label | None = Field(
        default=None,
        title="Label",
        description="Optional label filter. A multiclass model predicts all of its classes; if a label "
                    "is given, only instances of that label are returned.",
    )


class PromptedSegmentationRequest(BaseServiceRequest):
    """ Model for prompted segmentation. """
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: BinaryMask | None = Field(None, title="Previous Mask")


class InstanceSuggestionRequest(BaseServiceRequest):
    """ Model for instance suggestion with image exemplars and concepts. """
    positive_exemplars: list[BinaryMask] = Field(..., description="Exemplars is a list of RLE encoded binary masks")
    negative_exemplars: list[BinaryMask] | None = Field(None, title="Negative exemplars")
    concept: Label | None = Field(default=None, description="Optional label defining the concept.")

    @cached_property
    def positive_exemplar_masks(self) -> list[np.ndarray]:
        return [exemplar.mask for exemplar in self.positive_exemplars]

    @cached_property
    def negative_exemplar_masks(self) -> list[np.ndarray]:
        if self.negative_exemplars is None:
            return []
        return [exemplar.mask for exemplar in self.negative_exemplars]

    @cached_property
    def combined_exemplar_mask(self) -> np.ndarray:
        combined_mask = self.positive_exemplars[0].mask
        if len(self.positive_exemplars) > 1:
            for exemplar in self.positive_exemplars[1:]:
                combined_mask = np.logical_or(combined_mask, exemplar.mask)
        return combined_mask

    def get_bboxes(self,
                   format: Literal["xywh", "xyxy", "cxcywh"] = "xyxy",
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
            elif format == "xyxy":
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


class EmbedRegion(BaseModel):
    """One masked region to embed, tagged with an id the caller maps the result back to.

    ``region_id`` is opaque to the service -- typically a contour id -- and is echoed on the
    returned :class:`EmbeddingVector` so the caller can persist the vector against the right row.
    """

    region_id: int = Field(
        ...,
        description="Caller's id for this region (e.g. a contour id); echoed on the returned vector.",
    )
    mask: BinaryMask = Field(..., description="RLE-encoded binary mask selecting the region's foreground.")

    class Config:
        ignored_types = (cached_property,)

    @cached_property
    def region_mask(self) -> np.ndarray:
        """The decoded ``(H, W)`` boolean foreground mask."""
        return self.mask.mask


class EmbedRequest(BaseServiceRequest):
    """Request to precompute feature embeddings for an image and/or its masked regions.

    ``image_kinds`` are whole-image descriptors (e.g. ``"image_cls"``); each entry of
    ``regions`` yields one masked-region descriptor (``"region_mean"``). Both default to a
    single whole-image ``image_cls``; a caller may set either, but a request that asks for
    nothing (empty ``image_kinds`` and empty ``regions``) is a no-op. Which kinds a given
    embedder model actually understands is up to the model -- unknown kinds are skipped.
    """

    image_kinds: list[str] = Field(
        default_factory=lambda: ["image_cls"],
        description="Whole-image descriptor kinds to compute, e.g. ['image_cls']. May be empty.",
    )
    regions: list[EmbedRegion] = Field(
        default_factory=list,
        description="Masked regions to embed (each -> one 'region_mean' vector). May be empty.",
    )


class EmbeddingVector(BaseModel):
    """One computed embedding: the vector plus what it describes and which backbone made it.

    ``region_id`` is ``None`` for whole-image kinds and the region's id for region kinds.
    ``model_id`` is the concrete backbone (e.g. ``facebook/dinov3-vitb16-pretrain-lvd1689m``),
    not the registry key -- embeddings are only comparable within one ``model_id``, so the
    store persists it for versioning.
    """

    # ``model_id`` sits in pydantic's protected ``model_`` namespace; opt out so it is a plain
    # field (the store column is ``model_id``, so the name is worth keeping).
    model_config = {"protected_namespaces": ()}

    kind: str = Field(..., description="What the vector represents, e.g. 'image_cls' or 'region_mean'.")
    region_id: Optional[int] = Field(
        default=None, description="Region id for region kinds; None for whole-image kinds."
    )
    model_id: str = Field(..., description="The backbone that produced the vector (for store versioning).")
    dim: int = Field(..., description="Vector dimensionality.")
    vector: list[float] = Field(..., description="The L2-normalized embedding.")


class CrossImageExemplar(BaseModel):
    """One exemplar for cross-image concept transfer: an annotated object *in its own image*.

    Unlike :class:`InstanceSuggestionRequest`'s exemplars (masks on the request's single
    image), each cross-image exemplar carries its own ``image_url`` -- the retrieval layer
    picks exemplars from *other* images, and the concat handler needs each exemplar's pixels
    to paste beside the target.
    """

    image_url: str = Field(..., title="Exemplar image URL")
    mask: BinaryMask = Field(..., description="RLE-encoded binary mask of the exemplar object in its image.")

    class Config:
        ignored_types = (cached_property,)

    @cached_property
    def image(self) -> np.ndarray:
        """The exemplar's source image as an ``(H, W, 3)`` array."""
        return get_image_from_url_cached(self.image_url)

    @cached_property
    def exemplar_mask(self) -> np.ndarray:
        """The decoded ``(H, W)`` boolean object mask."""
        return self.mask.mask


class CrossImageSuggestionRequest(BaseServiceRequest):
    """Suggest instances of a concept on a target image, using exemplars from *other* images.

    SAM 3's prompted concept segmentation is intra-image only; this request drives the concat
    workaround -- the exemplar image(s) + mask(s) are composited beside the target so the model
    can transfer the concept across the seam. ``image_url`` (from the base) is the target being
    annotated; ``exemplars`` are the cross-image references (typically the top hits from the
    retrieval strategy); ``concept`` optionally adds a text prompt alongside the visual ones.
    """

    exemplars: list[CrossImageExemplar] = Field(
        default_factory=list, description="Cross-image exemplars (each an image + object mask)."
    )
    concept: Label | None = Field(default=None, description="Optional label adding a text prompt.")
