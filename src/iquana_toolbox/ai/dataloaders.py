"""
DataLoaders for instance segmentation tasks using torchvision and COCO datasets.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class COCOInstanceSegmentationDataset(CocoDetection):
    """
    PyTorch Dataset for COCO-style instance segmentation.

    Extends torchvision's CocoDetection to provide instance segmentation annotations
    (segmentation masks) for each image.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transforms_pipeline: Optional[Callable] = None,
    ):
        """
        Initialize COCO Instance Segmentation Dataset.

        Args:
            root (str): Root directory containing the images
            annFile (str): Path to the COCO annotation JSON file
            transforms_pipeline (Optional[Callable]): Optional transforms to apply to images
        """
        super().__init__(root=root, annFile=annFile)
        self.transforms_pipeline = transforms_pipeline

    def __getitem__(self, idx: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item

        Returns:
            tuple: (image, target) where target contains instance segmentation data
        """
        image, targets = super().__getitem__(idx)

        # Process targets to extract segmentation masks and bounding boxes
        processed_targets = self._process_targets(targets)

        if self.transforms_pipeline is not None:
            image = self.transforms_pipeline(image)

        return image, processed_targets

    def _process_targets(self, targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw COCO targets to extract segmentation information.

        Args:
            targets (List[Dict]): Raw COCO annotations

        Returns:
            Dict: Processed targets with masks, boxes, labels, and IDs
        """
        processed = {
            "boxes": [],
            "masks": [],
            "labels": [],
            "image_id": [],
            "iscrowd": [],
            "area": [],
        }

        for target in targets:
            # Extract bounding box
            if "bbox" in target:
                x, y, w, h = target["bbox"]
                box = [x, y, x + w, y + h]
                processed["boxes"].append(box)

            # Extract segmentation mask (RLE or polygon)
            if "segmentation" in target:
                processed["masks"].append(target["segmentation"])

            # Extract category label
            if "category_id" in target:
                processed["labels"].append(target["category_id"])

            # Extract image ID
            if "image_id" in target:
                processed["image_id"].append(target["image_id"])

            # Extract iscrowd flag (important for evaluation metrics)
            if "iscrowd" in target:
                processed["iscrowd"].append(target["iscrowd"])

            # Extract area
            if "area" in target:
                processed["area"].append(target["area"])

        # Convert to tensors where appropriate
        if processed["boxes"]:
            processed["boxes"] = torch.as_tensor(
                processed["boxes"], dtype=torch.float32
            )
        else:
            processed["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        if processed["labels"]:
            processed["labels"] = torch.as_tensor(
                processed["labels"], dtype=torch.int64
            )
        else:
            processed["labels"] = torch.zeros((0,), dtype=torch.int64)

        if processed["iscrowd"]:
            processed["iscrowd"] = torch.as_tensor(
                processed["iscrowd"], dtype=torch.uint8
            )
        else:
            processed["iscrowd"] = torch.zeros((0,), dtype=torch.uint8)

        if processed["area"]:
            processed["area"] = torch.as_tensor(
                processed["area"], dtype=torch.float32
            )
        else:
            processed["area"] = torch.zeros((0,), dtype=torch.float32)

        processed["image_id"] = processed["image_id"]
        processed["masks"] = processed["masks"]

        return processed


def get_coco_instance_segmentation_dataset(
    image_folder: str,
    annotation_file: str,
    transforms_pipeline: Optional[Callable] = None,
    **kwargs
) -> COCOInstanceSegmentationDataset:
    """
    Create and return a COCO instance segmentation dataset.

    This is a convenience function that creates a COCOInstanceSegmentationDataset
    with COCO-style annotations.

    Args:
        image_folder (str): Path to folder containing images
        annotation_file (str): Path to COCO annotation JSON file
        transforms_pipeline (Optional[Callable]): Optional transforms to apply
        **kwargs: Additional arguments passed to the dataset

    Returns:
        COCOInstanceSegmentationDataset: PyTorch dataset for instance segmentation

    Example:
        >>> dataset = get_coco_instance_segmentation_dataset(
        ...     image_folder='/path/to/images',
        ...     annotation_file='/path/to/annotations.json'
        ... )
        >>> image, target = dataset[0]
    """
    return COCOInstanceSegmentationDataset(
        root=image_folder,
        annFile=annotation_file,
        transforms_pipeline=transforms_pipeline,
    )


def get_coco_instance_segmentation_dataloader(
    image_folder: str,
    annotation_file: str,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    transforms_pipeline: Optional[Callable] = None,
    **kwargs
) -> DataLoader:
    """
    Create and return a DataLoader for COCO instance segmentation.

    Args:
        image_folder (str): Path to folder containing images
        annotation_file (str): Path to COCO annotation JSON file
        batch_size (int): Batch size for the dataloader. Default: 8
        num_workers (int): Number of worker processes. Default: 0
        shuffle (bool): Whether to shuffle the dataset. Default: True
        transforms_pipeline (Optional[Callable]): Optional transforms to apply
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        DataLoader: PyTorch DataLoader for instance segmentation

    Example:
        >>> dataloader = get_coco_instance_segmentation_dataloader(
        ...     image_folder='/path/to/images',
        ...     annotation_file='/path/to/annotations.json',
        ...     batch_size=16,
        ...     num_workers=4
        ... )
        >>> for images, targets in dataloader:
        ...     pass
    """
    dataset = get_coco_instance_segmentation_dataset(
        image_folder=image_folder,
        annotation_file=annotation_file,
        transforms_pipeline=transforms_pipeline,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs
    )

    return dataloader

