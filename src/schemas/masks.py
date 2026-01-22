from functools import cached_property

import numpy as np
from pycocotools import mask as maskUtils

from pydantic import BaseModel, Field, computed_field

from src.schemas.labels import LabelHierarchy


class BinaryMask(BaseModel):
    rle_mask: dict = Field(..., description="Binary mask RLE encoded")
    score: float | None = Field(..., description="A quality score of the mask")
    height: int = Field(..., description="The height of the mask")
    width: int = Field(..., description="The width of the mask")

    class Config:
        # This allows property to work smoothly with Pydantic
        ignored_types = (cached_property,)

    @cached_property
    def mask(self):
        return maskUtils.decode(self.rle_mask)

    @classmethod
    def from_numpy_array(cls, binary_mask: np.ndarray, score=None):
        # As fortranarray
        binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        # Encode to RLE
        encoded = maskUtils.encode(binary_mask)
        # Convert bytes to string so it's JSON serializable
        encoded['counts'] = encoded['counts'].decode('utf-8')
        return cls(
            rle_mask=encoded,
            score=score,
            height=binary_mask.shape[0],
            width=binary_mask.shape[1],
        )


class SemanticMask(BaseModel):
    label_id_to_binary_mask: dict[int, BinaryMask] = Field(..., description="Dictionary mapping from label ids to "
                                                                            "Binary mask objects. ")
    label_hierarchy: LabelHierarchy = Field(..., description="Label hierarchy used to get label values and label names.")

    class Config:
        # This allows property to work smoothly with Pydantic
        ignored_types = (cached_property,)

    @computed_field
    def score(self):
        return np.sum(bin_mask.score for bin_mask in self.label_id_to_binary_mask.values())

    @computed_field
    def height(self):
        return (self.label_id_to_binary_mask.values())[0].height

    @computed_field
    def width(self):
        return (self.label_id_to_binary_mask.values())[0].width

    @cached_property
    def mask(self):
        """
        Converts to standard numpy array with dimensions height and width. The entry values correspond to the label.
        """
        sem_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for label_id, binary_mask in self.label_id_to_binary_mask.items():
            label_value = self.label_hierarchy.id_to_label_object[label_id].value
            sem_mask[binary_mask.mask] = label_value
        return sem_mask

    @cached_property
    def one_hot_mask(self):
        """ Converts to one-hot encoding. N_labels x height x width numpy array, where each subarray is binary"""
        max_label_value = np.max(self.label_hierarchy.value_to_label_object.keys()).item()
        oh_mask = np.zeros((max_label_value, self.height, self.width), dtype=np.uint8)
        for label_id, binary_mask in self.label_id_to_binary_mask.items():
            label_value = self.label_hierarchy.id_to_label_object[label_id].value
            oh_mask[label_value] = binary_mask.mask
        return oh_mask

    @classmethod
    def from_numpy_array(cls, numpy_array: np.ndarray, label_hierarchy: LabelHierarchy):
        """
        Instantiate from a numpy array and label hierarchy
        :param numpy_array: height x width numpy array with entry values corresponding to the label
        :param label_hierarchy: LabelHierachy object.
        :return: Instance of this class.
        """
        uniques = np.unique(numpy_array)
        label_id_to_binary_mask = {}
        for unique_value in uniques:
            if unique_value == 0:
                # Skip background
                continue
            bin_array = numpy_array == unique_value
            label_id = LabelHierarchy.value_to_label_object[unique_value].id
            label_id_to_binary_mask[label_id] = BinaryMask.from_numpy_array(bin_array)
        return cls(
            label_hierarchy=label_hierarchy,
            label_id_to_binary_mask=label_id_to_binary_mask,
        )
