"""MNIST dataset module."""
import os
from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from mmpie1 import Config
from mmpie1.utils import ifnone

from mmpie1.data.transforms import Augmentations
from mmpie1.data.preprocessor import Preprocessor, resize_and_pad_to_target, resize_boxes
from mmpie1.data.COCODetectionDataset import CocoDetection

from omegaconf import DictConfig


def collate_fn(batch):
    max_height = max([item['pixel_values'].shape[1] for item in batch])
    max_width = max([item['pixel_values'].shape[2] for item in batch])
    target_size = (max_height, max_width)

    pixel_values = []
    boxes = []
    labels = []
    pixel_masks = []

    for item in batch:
        image = item['pixel_values']
        mask = item['pixel_mask']
        bboxes = item['boxes']

        
        resized_image, scale, padding = resize_and_pad_to_target(image, target_size)
        resized_mask, _, _ = resize_and_pad_to_target(mask.unsqueeze(0), target_size)
        resized_mask = resized_mask.squeeze(0)

        
        if bboxes.size(0) == 0:
            resized_boxes = torch.zeros((0, 4), dtype=torch.float32)
            item_labels = torch.full((0,), -1, dtype=torch.int64)  
        else:
            resized_boxes = resize_boxes(bboxes.clone(), scale, padding)
            item_labels = item['labels']
        
        pixel_values.append(resized_image)
        pixel_masks.append(resized_mask)
        boxes.append(resized_boxes)
        labels.append(item_labels)

    pixel_values = torch.stack(pixel_values)
    pixel_masks = torch.stack(pixel_masks)

    batch_dict = {
        'pixel_values': pixel_values,
        'boxes': boxes,
        'labels': labels,
        'pixel_masks': pixel_masks,
        'original_images': None,
        'original_boxes': None,
        'profiling': [item['profiling'] for item in batch]
    }
    
    return batch_dict





class MMPieDataModule(pl.LightningDataModule):
    """MNIST dataset module.

    The MNISTDataModule is a PyTorch Lightning DataModule which provides train, val, and test dataloaders for the
    MNIST dataset, and may be passed directly into a Pytorch Lightning Trainer. The MNIST dataset is a collection of
    70,000 28x28 grayscale images of handwritten digits (0-9) and their corresponding labels. The dataset is split into
    60,000 training images and 10,000 test images. The training images are further commonly split into 55,000 training
    and 5,000 validation images, although these numbers may be set by the user.

    Example, Manually Preparing and Using the DataModule::

        from mmpie1.data import MNIST

        # Instantiate MNIST dataset module
        dm = MNIST()
        dm.prepare_data()  # downloads data
        dm.setup()  # splits into train/val/test

        # Torch DataLoaders can then be accessed with:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

    Example, Using the DataModule with a PyTorch Lightning Trainer::

        from mmpie1.data import MNIST
        from pytorch_lightning import Trainer

        # Instantiate MNIST dataset module
        dm = MNIST()

        # Pass MNIST dataset module to PyTorch Lightning Trainer
        trainer = Trainer()
        trainer.fit(model, dm)

    """

    config = Config()

    def __init__(
        self,
        cfg: DictConfig,
        BatchSize: int = 8,
        NumWorkers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.batch_size = cfg.BatchSize
        # self.num_workers = num_workers if num_workers is not None else max(os.cpu_count() // 2, 1)
        self.num_workers = cfg.NumWorkers
        self.pin_memory = pin_memory

        self.cfg = cfg

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.prepare_data()
        self.setup()



    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.augmentations = Augmentations(**self.cfg.Augmentations)
        self.preprocessing = Preprocessor(**self.cfg.Preprocessing)

        self.train_dataset = CocoDetection(self.cfg.TrainPath, 
                                           augmentations = self.augmentations, 
                                           preprocessing = self.preprocessing, 
                                           train = True, profiling_req = False, return_original=False)
        
        self.val_dataset = CocoDetection(self.cfg.ValPath, 
                                         augmentations = None, 
                                         preprocessing = self.preprocessing, 
                                         train = False,  profiling_req = False, return_original=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory,  collate_fn=collate_fn
        )


    def sample(self, stage: str = "train", idx: int = 0):
        """Return a sample from the dataset.

        Args:
            stage: The stage of the dataset from which to sample. Must be one of: train, val, test.
            idx: The index of the sample to return.

        Returns:
            A sample from the dataset as an (image, label) tuple.
        """
        if stage == "train":
            return self.train_dataset[idx]
        elif stage == "val":
            return self.val_dataset[idx]
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be one of: train, val.")
        


