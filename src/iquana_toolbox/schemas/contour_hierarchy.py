from collections import defaultdict, deque

import cv2
import numpy as np
from pydantic import BaseModel, Field

from iquana_toolbox.schemas.contours import Contour, logger
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.contours import get_contours_from_binary_mask


def logical_or_contours(contours: list[Contour], shape: tuple[int, int]) -> np.ndarray:
    new_mask = np.zeros(shape, dtype=bool)
    if len(contours) == 0:
        return new_mask
    for contour in contours:
        bin_mask = contour.to_binary_mask(
            height=new_mask.shape[1],
            width=new_mask.shape[0]
        )
        new_mask = np.logical_or(new_mask, bin_mask)
    return new_mask


class ContourHierarchy(BaseModel):
    """ A hierarchy of contours. """
    root_contours: list[Contour] = Field(default_factory=list, description="List of objects represented by their contours.")
    id_to_contour: dict[int, Contour] = Field(
        default_factory=dict,
        description="Dict mapping contour id to object."
    )
    label_id_to_contours: dict[int | None, list[Contour]] = Field(
        default_factory=dict,
        description="Dict mapping label id to a list of objects."
    )

    @classmethod
    def from_query(cls, all_db_entries, width, height) -> "ContourHierarchy":
        """ Adds all contours in a breadth first search, then connects them to a hierarchy. """
        if not all_db_entries:
            return cls()

        # Map DB entries to our Pydantic Contour objects
        # We create a dict for fast O(1) lookup by ID
        id_to_contour = {}
        label_id_to_contours = defaultdict(list)

        for entry in all_db_entries:
            contour_obj = Contour.from_db(entry, width, height)
            id_to_contour[entry.id] = contour_obj
            label_id_to_contours[entry.label_id].append(contour_obj)

        # Build the tree structure in memory
        root_contours = []

        for entry in all_db_entries:
            current_obj = id_to_contour[entry.id]

            if entry.parent_id is None:
                # This is a top-level object (e.g., a Cell)
                root_contours.append(current_obj)
            else:
                # This is a child (e.g., a Nucleus inside a Cell)
                # Connect it to its parent object in memory
                parent_obj = id_to_contour.get(entry.parent_id)
                if parent_obj:
                    parent_obj.add_child(current_obj)
                else:
                    # Fallback: if parent isn't in query, treat as root
                    root_contours.append(current_obj)
        return cls(
            root_contours=root_contours,
            id_to_contour=id_to_contour,
            label_id_to_contours=label_id_to_contours,
        )

    def get_children_mask(self, contour_id, shape):
        if contour_id is None:
            # The children of a contour without a parent are the root contours
            contours = self.root_contours
        else:
            contours = self.id_to_contour[contour_id].children
        return logical_or_contours(
            contours,
            shape
        )

    def get_parent_mask(self, parent_id, shape):
        if parent_id is None:
            return np.ones(shape, dtype=np.uint8)
        else:
            return self.id_to_contour[parent_id].to_binary_mask(shape[0], shape[1])

    def add_contour(self, contour: Contour):
        # Get a binary mask indicating which pixels can be inside the contour
        # Gets all contours on the same level
        resolution_shape = (1000, 1000)
        allowed_pixels_level = np.logical_not(
            self.get_children_mask(contour.parent_id, resolution_shape)
        )
        # Get the parent contour. New contour must be inside it!
        allowed_pixels = np.logical_and(
            allowed_pixels_level,
            self.get_parent_mask(contour.parent_id, resolution_shape)
        )
        print(f"Allowed pixels on level: {np.sum(allowed_pixels)}")
        print(f"Allowed pixels from parent: {np.sum(self.get_parent_mask(contour.parent_id, resolution_shape))}")
        print(f"Allowed pixels after fitting: {np.sum(allowed_pixels)}")
        if not np.any(allowed_pixels):
            print("Could not add contour! No pixels after fitting to parents and other contours!")

        contour, changed = contour.fit_to_mask(allowed_pixels)
        if contour.parent_id is None:
            # We add a root contour
            self.root_contours.append(contour)
            self.id_to_contour[contour.id] = contour
            self.label_id_to_contours.setdefault(contour.label_id, []).append(contour)
        else:
            # We add a child contour
            self.id_to_contour[contour.parent_id].add_child(contour)
            self.id_to_contour[contour.id] = contour
            self.label_id_to_contours.setdefault(contour.label_id, []).append(contour)
        return contour, changed

    def dump_contours_as_list(self, breadth_first: bool = True) -> list[Contour]:
        """ Dump all contours in the hierarchy as a list. Can be done in breadth first or depth first order. """
        contours_list = []
        queue = deque(self.root_contours)
        while queue:
            contour = queue.popleft()
            contours_list.append(contour)
            if breadth_first:
                queue.extend(contour.children)
            else:
                queue.extendleft(reversed(contour.children))
        return contours_list

    @classmethod
    def from_semantic_mask(cls,
                           np_mask: np.ndarray,
                           label_hierarchy: LabelHierarchy,
                           added_by: str, ):
        """
        Get a contour hierarchy from a mask and a label hierarchy. The hierarchy will respect both the label
        hierarchy as well as spatial hierarchy, i.e. each child contour lies within its parent.
        """
        contour_models_with_label_id = {}
        root_contours = []
        id_to_contour = {}
        label_id_to_contours = defaultdict(list)
        flat_label_hierarchy = label_hierarchy.build_flat_hierarchy(breadth_first=True)
        height, width = np_mask.shape[:2]
        for label in flat_label_hierarchy:
            # Go through the labels by a breadth first search
            if label.value == 0:
                # Skip the background label (usually 0)
                continue

            # First: Extract the mask for the current label and create Contour Models
            mask_label = (np_mask == label.value).astype(np.uint8)
            contour_models = get_contours_from_binary_mask(mask_label,
                                                           only_return_biggest=False,
                                                           limit=None,
                                                           added_by=added_by,
                                                           label_id=label.id,
                                                           )

            contour_entries = []

            # Second: Iterate through the models and check for parent links
            parent = label_hierarchy.get_parent_by_value_of_child(label.value)
            if parent is not None:
                for contour, entry in zip(contour_models, contour_entries):
                    # For each contour, that we found, we check:
                    for parent_contour in label_id_to_contours[parent.value]:
                        # Does any parent label contour exist, in which the contour lies
                        # Depending on the nesting, this can take quite a while
                        if contour in parent_contour:
                            contour.parent_id = parent_contour.id
                            entry.parent_id = parent_contour.id
                            parent_contour.add_child(contour)
                            break
                    else:
                        # This should not happen, something is wrong.
                        logger.error("Contour could not be added to a parent contour")
            else:
                root_contours.extend(contour_models)
            contour_models_with_label_id[label] = contour_models
        return cls(
            root_contours=root_contours,
            id_to_contour=id_to_contour,
            label_id_to_contours=label_id_to_contours,
        )

    def to_semantic_mask(self, height, width, label_id_to_value_map: dict[int, int]) -> np.ndarray:
        """ Turn the hierarchy into a semantic mask of the given shape. In a semantic mask each pixel value represents a
        class. """
        # Create empty canvas
        canvas = np.zeros((height, width), dtype=np.uint8)
        # Create empty queue
        queue = deque()
        # Enqueue root contours
        queue.extend(self.root_contours)
        while queue:
            # Remove the oldest entry
            contour = queue.popleft()
            # If the contour has no label we cannot add it to the mask
            if contour.label_id:
                canvas = cv2.drawContours(canvas,
                                          contour.to_rescaled_contour(height, width),
                                          -1,  # -1 means fill the contour
                                          [label_id_to_value_map[contour.label_id]],
                                          1)
                # Add all children to the queue
                if len(contour.children) > 0:
                    queue.extend(contour.children)
        # Return the filled array
        return canvas

    def get_label_quantification(self, label_id):
        # First get all relevant contours
        contours = self.label_id_to_contours.get(label_id, [])

        # Second track all relevant metrics
        metrics = defaultdict(list)
        child_counts = defaultdict(list)
        for contour in contours:
            # Get the quantifications from the quantification model
            for quant_key, quant_value in contour.quantification.model_dump().items():
                metrics[quant_key].append(quant_value)

            # Count the children
            _child_counts = defaultdict(lambda: 0)
            for child_contour in contour.children:
                _child_counts[child_contour.label_id] += 1

            for label_id in _child_counts:
                child_counts[label_id].append(_child_counts[label_id])
        return {
            "metrics": metrics,
            "child_counts": child_counts,
        }

    def get_all_quantifications(self):
        response = {}
        for label in self.label_id_to_contours.keys():
            response[label] = self.get_label_quantification(label)
        return response
