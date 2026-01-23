import json
from logging import getLogger
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, field_validator, Field, model_validator
from pycocotools import mask as maskUtils

from .masks import BinaryMask
from .quantification import QuantificationModel

logger = getLogger(__name__)


def get_contours(mask,
                 retr_str: int = cv2.RETR_EXTERNAL,
                 approx_str: int = cv2.CHAIN_APPROX_SIMPLE,
                 normalized: bool = True,):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, retr_str, approx_str)
    # Convert contours to a numpy array, otherwise is sequence of np arrays. Code below wouldnt run.
    contours = np.array(contours, dtype=np.int32)
    logger.debug(f"Found {len(contours)} contours.")
    if normalized:
        # Normalize contours
        contours[..., 0] = contours[..., 0] / mask.shape[1]
        contours[..., 1] = contours[..., 1] / mask.shape[0]
    return contours, hierarchy


class Contour(BaseModel):
    """ Model for a contour to be added. """
    id: int | None = Field(default=None, description="Contour id. Only pass None if the id is not yet known.")
    label_id: int | None = Field(default=None, description="ID of the label of the mask. None for unlabelled contour.")
    parent_id: int | None = Field(default=None, description="ID of the parent contour. None if the contour has "
                                                                    "no parent")
    children: list["Contour"] = Field(default=[], description="List of objects represented by their contours.")
    reviewed_by: list[str] = Field(default=[], description="List of users who reviewed the contour.")

    x: list[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: list[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    path: str | None = Field(default=None, description="SVG path string for rendering the contour.")

    added_by: str = Field(default_factory=str, description="ID of the user or model who added this contour.")
    confidence: float = Field(default=1., description="Confidence score of the contour.")
    quantification: QuantificationModel | None = Field(default=None, description="Quantification of the contour. Does "
                                                                                 "not need to be provided.")

    @model_validator(mode="after")
    def validate_after(self):
        """ Validate after initialization. Check if quantifications are computed. """
        if self.quantification is None:
            self.quantification = QuantificationModel.from_cv_contour(self.contour)
        elif self.quantification.is_empty:
            self.quantification.parse_cv_contour(self.contour)
        return self

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0.), 1.) for coord in value]

    @property
    def contour(self) -> np.ndarray:
        """
        As a opencv contour.
        Opencv contours have the form Number of points x empty dimension x Tuple of x and y coordinate.
        """
        return np.expand_dims(self.points, axis=1)

    @property
    def points(self) -> np.ndarray[tuple[float, float]]:
        return np.array(list(zip(self.x, self.y)))

    def get_children_by_label(self, label_id):
        children = []
        for child in self.children:
            if child.label_id == label_id:
                children.append(child)
        return children

    def compute_path(self, image_width: int, image_height: int):
        """Compute SVG path from normalized coordinates (0-1) to pixel coordinates."""
        if not self.x or not self.y or len(self.x) == 0:
            self.path = ""
            return
        first_x = round(self.x[0] * image_width)
        first_y = round(self.y[0] * image_height)
        path = f"M {first_x} {first_y}"
        for i in range(1, len(self.x)):
            x = round(self.x[i] * image_width)
            y = round(self.y[i] * image_height)
            path += f" L {x} {y}"
        self.path = path + " Z"

    def to_binary_mask(self, height, width) -> np.ndarray:
        """ Return a binary mask given the height and width. """
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        binary_mask = cv2.drawContours(
            image=binary_mask,
            contours=[self.to_rescaled_contour(height, width)],
            contourIdx=-1,  # -1 means fill the contour
            color=1,
            thickness=cv2.FILLED
        )
        return binary_mask.astype(bool)

    def to_rle_encoding(self, height, width):
        bin_mask = self.to_binary_mask(height, width)
        return maskUtils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))

    def to_rescaled_contour(self, height, width):
        """ Return a rescaled contour given the height and width. """
        rescaled_x = (np.array(self.x) * width).astype(int)
        rescaled_y = (np.array(self.y) * height).astype(int)
        return np.expand_dims(np.array(list(zip(rescaled_x, rescaled_y))), axis=1)

    @classmethod
    def from_normalized_cv_contour(cls, normalized_cv_contour, label_id, added_by):
        x_coords = normalized_cv_contour[..., 0].flatten()
        y_coords = normalized_cv_contour[..., 1].flatten()
        return cls(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label_id=label_id,
            added_by=added_by,
        )

    @classmethod
    def from_db(cls, contour, image_width: int | None = None, image_height: int | None = None):
        contour_obj = cls(
            id=contour.id,
            parent_id=contour.parent_id,
            label_id=contour.label_id,
            x=json.loads(contour.x) if type(contour.x) is str else contour.x,
            y=json.loads(contour.y) if type(contour.y) is str else contour.y,
            added_by=contour.added_by,
            reviewed_by=[user.username for user in contour.reviewed_by],
            quantification=QuantificationModel(
                area=contour.area,
                perimeter=contour.perimeter,
                circularity=contour.circularity,
                max_diameter=contour.diameter,
            )
        )
        # Compute SVG path if image dimensions are provided
        if image_width is not None and image_height is not None:
            contour_obj.compute_path(image_width, image_height)
        return contour_obj

    @classmethod
    def from_binary_mask(cls,
                         binary_mask: np.ndarray,
                         only_return_biggest_contour: bool = True,
                         **kwargs):
        """
            Convert a binary mask into a contour list.
            :param binary_mask: The mask to turn into a list of contours.
            :param only_return_biggest_contour: If true, returns only the biggest contour, not a list!
            :param kwargs: Additional keyword arguments. All contours will be initialized with these.
            :returns: List of contour models.
        """
        contours, _ = get_contours(binary_mask)

        if only_return_biggest_contour:
            contour = max(contours, key=cv2.contourArea).astype(float)
            return cls.from_normalized_cv_contour(
                normalized_cv_contour=contour,
                **kwargs
            )
        else:
            return [cls.from_normalized_cv_contour(
                normalized_cv_contour=contour,
                **kwargs
            ) for contour in contours]

    @classmethod
    def from_binary_mask_model(cls,
                               mask_model: BinaryMask,
                               **kwargs):
        """ Create a contour from a binary mask model """
        return cls.from_binary_mask(
            binary_mask=mask_model.mask,
            confidence=mask_model.score,
            **kwargs
        )

    def __in__(self, other):
        """
        Is this contour contained in another contour. Checks whether all points in this contour are inside another.
        """
        return all(cv2.pointPolygonTest(
            (other.contour * 10_000).astype(np.uint32),
            pt=(pt * 10_000).astype(np.uint32),
            measureDist=False) >= 0 for pt in self.points)

    def add_child(self, child: "Contour"):
        # Prevent duplicates and set the parent-child link
        if child not in self.children:
            self.children.append(child)


def get_contours_from_binary_mask(mask: np.ndarray,
                                  only_return_biggest=False,
                                  limit=None,
                                  added_by: str = "system",
                                  label_id: int = None,) -> list[Contour]:
    """ Get contour models from a binary mask
    :param mask: A binary mask in the form of a numpy array
    :param only_return_biggest: If true, only return the biggest contour.
    :param limit: Number of contours to return. If None, return all contours.
    :param added_by: Author of this contour, by default "system".
    :param label_id: Contour label id. If None, no label is given to the contour.
    :return: List of contour models
    """
    logger.debug("Computing contours for mask.")
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # check if any contours found
        logger.info(f"Found {len(contours)} contours.")
        if only_return_biggest:
            contours = [max(contours, key=cv2.contourArea)]
        else:
            contours = sorted(contours, key=cv2.contourArea)
            if limit is not None and len(contours) > limit:
                logger.warning(f"Detected over {limit} objects. Only returning the biggest 500 objects.")
                contours = contours[:limit]
        models = []
        for contour in contours:
            # Skip one dimensional contours
            if contour.shape[0] <= 2:
                continue
            # First dim of contour is x, but first dim of mask is height, so it needs to be switched!
            contour = contour.astype(float)
            contour[..., 0] /= mask.shape[1]
            contour[..., 1] /= mask.shape[0]
            models.append(Contour.from_normalized_cv_contour(contour,
                                                             label_id=label_id,
                                                             added_by=added_by)
                          )
        return models
    else:
        logger.info(f"No contours found for mask: {mask}")
        return np.array([])
