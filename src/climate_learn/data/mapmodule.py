"""
Map-style data module for climate data downscaling tasks.
"""

import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .npzdataset import NpzDataset


def collate_fn(batch):
    """
    Collate function for batching data samples with padding.
    
    Adds padding to output tensors to match target resolution:
    - 2 pixels on left/right (longitude)
    - 3 pixels on top/bottom (latitude)
    
    Args:
        batch: List of (input, output) tuples
        
    Returns:
        Tuple of (inputs, padded_outputs, input_vars, output_vars)
    """
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    out = F.pad(out, (2, 2, 3, 3))  # Pad for resolution matching
    return inp, out, ["daily_tmax"], ["daily_tmax"]


class ERA5toPRISMDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ERA5 to PRISM downscaling.
    
    This module handles the loading and preprocessing of climate data for 
    spatial downscaling tasks, specifically from ERA5 resolution to higher
    resolution PRISM data.
    
    Attributes:
        in_root_dir: Directory containing ERA5 (input) data
        out_root_dir: Directory containing PRISM (output) data
        batch_size: Number of samples per batch
        num_workers: Number of data loading worker processes
    """

    def __init__(self, in_root_dir, out_root_dir, batch_size=32, num_workers=4):
        """
        Initialize the data module.
        
        Args:
            in_root_dir: Path to ERA5 data directory
            out_root_dir: Path to PRISM data directory
            batch_size: Number of samples per batch (default: 32)
            num_workers: Number of data loading workers (default: 4)
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.hparams.out_vars = ["daily_tmax"]  # Maximum daily temperature
        self.hparams.history = 1  # Single timestep for spatial downscaling
        self.hparams.task = "downscaling"

    def setup(self, stage="foobar"):
        """
        Set up datasets for training, validation, and testing.
        
        Loads datasets and their associated transforms, coordinates, and masks.
        The transforms are shared across splits to ensure consistent normalization.
        
        Args:
            stage: Unused stage parameter (kept for compatibility)
        """
        # Initialize training dataset and get transforms
        self.train_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "train.npz"),
            os.path.join(self.hparams.out_root_dir, "train.npz"),
        )
        self.in_transform = self.train_dataset.in_transform
        self.out_transform = self.train_dataset.out_transform

        # Initialize validation and test datasets with same transforms
        self.val_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "val.npz"),
            os.path.join(self.hparams.out_root_dir, "val.npz"),
            self.in_transform,
            self.out_transform,
        )
        self.test_dataset = NpzDataset(
            os.path.join(self.hparams.in_root_dir, "test.npz"),
            os.path.join(self.hparams.out_root_dir, "test.npz"),
            self.in_transform,
            self.out_transform,
        )

        # Load mask and coordinate information
        self.out_mask = torch.from_numpy(
            np.load(os.path.join(self.hparams.out_root_dir, "mask.npy"))
        )
        
        # Load input (ERA5) coordinates
        with open(os.path.join(self.hparams.in_root_dir, "coords.npz"), "rb") as f:
            npz = np.load(f)
            self.in_lat = torch.from_numpy(npz["lat"])
            self.in_lon = torch.from_numpy(npz["lon"])
            
        # Load output (PRISM) coordinates
        with open(os.path.join(self.hparams.out_root_dir, "coords.npz"), "rb") as f:
            npz = np.load(f)
            self.out_lat = torch.from_numpy(npz["lat"])
            self.out_lon = torch.from_numpy(npz["lon"])

    def get_lat_lon(self):
        """Get output (PRISM) latitude and longitude coordinates."""
        return self.out_lat, self.out_lon

    def get_data_dims(self):
        """
        Get input and output tensor dimensions.
        
        Returns:
            Tuple of (input_shape, output_shape) including padding
        """
        x, y = self.train_dataset[0]
        y = F.pad(y, (2, 2, 3, 3))  # Add padding to output
        return x.unsqueeze(0).shape, y.unsqueeze(0).shape

    def get_data_variables(self):
        """Get lists of input and output variables."""
        return ["daily_tmax"], ["daily_tmax"]

    def get_climatology(self, split):
        """
        Get climatological mean for specified data split.
        
        Args:
            split: One of 'train', 'val', or 'test'
            
        Returns:
            Per-pixel mean values for the specified split
        """
        if split == "train":
            return self.train_dataset.out_per_pixel_mean
        elif split == "val":
            return self.val_dataset.out_per_pixel_mean
        elif split == "test":
            return self.test_dataset.out_per_pixel_mean
        else:
            raise NotImplementedError()

    def get_out_transforms(self):
        """Get output data transforms."""
        return self.out_transform

    def get_out_mask(self):
        """Get padded output mask."""
        padded_mask = F.pad(self.out_mask, (2, 2, 3, 3))
        return padded_mask

    # DataLoader methods with consistent configuration
    def train_dataloader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_fn,
        )
