# Contains IterDataModule, which is a PyTorch Lightning DataModule for climate data processing.

# Standard library
import copy
import glob
import os
from typing import Dict, Optional, Union, List

# Third party
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import transforms
import pytorch_lightning as pl

# Local application
from .iterdataset import (
    NpyReader,
    DirectForecast,
    ContinuousForecast,
    Downscale,
    IndividualDataIter,
    ShuffleIterableDataset,
)


class IterDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for climate data processing.
    
    This module handles:
    1. Loading and preprocessing climate data for different tasks
    2. Setting up train/val/test data loaders
    3. Managing data transformations and batching
    4. Supporting different prediction tasks:
       - Direct forecasting (single time step)
       - Iterative forecasting (multiple time steps)
       - Continuous forecasting (variable lead times)
       - Downscaling (spatial resolution enhancement)
    
    Attributes:
        task (str): Type of prediction task
        inp_root_dir (str): Directory containing input data
        out_root_dir (str): Directory containing output/target data
        in_vars (list): Input variables to use
        out_vars (list): Output variables to predict
        src (Optional[str]): Source dataset identifier
        history (int): Number of historical time steps to use
        window (int): Window size for temporal processing
        pred_range (int): Prediction horizon (in time steps)
        random_lead_time (bool): Whether to use random lead times
        max_pred_range (int): Maximum prediction horizon
        hrs_each_step (int): Hours between time steps
        subsample (int): Subsampling factor for data
        buffer_size (int): Size of shuffle buffer
        batch_size (int): Number of samples per batch
        num_workers (int): Number of data loading workers
        pin_memory (bool): Whether to pin memory in data loading
    """

    def __init__(
        self,
        task,
        inp_root_dir,
        out_root_dir,
        in_vars,
        out_vars,
        src=None,
        history=1,
        window=6,
        pred_range=6,
        random_lead_time=True,
        max_pred_range=120,
        hrs_each_step=1,
        subsample=1,
        buffer_size=10000,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Configure dataset class and arguments based on task
        if task in ("direct-forecasting", "iterative-forecasting"):
            self.dataset_caller = DirectForecast
            self.dataset_arg = {
                "src": src,
                "pred_range": pred_range,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn
        elif task == "continuous-forecasting":
            self.dataset_caller = ContinuousForecast
            self.dataset_arg = {
                "random_lead_time": random_lead_time,
                "min_pred_range": pred_range,
                "max_pred_range": max_pred_range,
                "hrs_each_step": hrs_each_step,
                "history": history,
                "window": window,
            }
            self.collate_fn = collate_fn_continuous
        elif task == "downscaling":
            raise NotImplementedError("Downscaling is not implemented yet")
            # self.dataset_caller = Downscale
            # self.dataset_arg = {}
            # self.collate_fn = collate_fn

        # make sure the data in the dataset folders are sorted
        self.inp_file_list_train = sorted(
            glob.glob(os.path.join(inp_root_dir, "train", "*.npz"))
        )
        self.out_file_list_train = sorted(
            glob.glob(os.path.join(out_root_dir, "train", "*.npz"))
        )
        self.inp_file_list_val = sorted(
            glob.glob(os.path.join(inp_root_dir, "val", "*.npz"))
        )
        self.out_file_list_val = sorted(
            glob.glob(os.path.join(out_root_dir, "val", "*.npz"))
        )
        self.inp_file_list_test = sorted(
            glob.glob(os.path.join(inp_root_dir, "test", "*.npz"))
        )
        self.out_file_list_test = sorted(
            glob.glob(os.path.join(out_root_dir, "test", "*.npz"))
        )
        
        # Check which datasets exist
        self.has_train_files = len(self.inp_file_list_train) > 0 and len(self.out_file_list_train) > 0
        self.has_val_files = len(self.inp_file_list_val) > 0 and len(self.out_file_list_val) > 0
        self.has_test_files = len(self.inp_file_list_test) > 0 and len(self.out_file_list_test) > 0

        self.transforms = self.get_normalize(inp_root_dir, in_vars)
        self.output_transforms = self.get_normalize(out_root_dir, out_vars)

        # Initialize datasets
        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None
        

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.out_root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.out_root_dir, "lon.npy"))
        return lat, lon

    def get_data_variables(self):
        return self.hparams.in_vars, self.hparams.out_vars

    def get_data_dims(self):
        lat = len(np.load(os.path.join(self.hparams.out_root_dir, "lat.npy")))
        lon = len(np.load(os.path.join(self.hparams.out_root_dir, "lon.npy")))
        forecasting_tasks = [
            "direct-forecasting",
            "iterative-forecasting",
            "continuous-forecasting",
        ]
        # Shape: [batch, time_steps, variables, lat, lon] for forecasting
        if self.hparams.task in forecasting_tasks:
            in_size = torch.Size(
                [
                    self.hparams.batch_size,
                    self.hparams.history,
                    len(self.hparams.in_vars),
                    lat,
                    lon,
                ]
            )
        else:
            raise ValueError(
                f"Unsupported task type: '{self.hparams.task}'. "
                f"Supported tasks are: {forecasting_tasks}"
            )
        out_size = torch.Size([self.hparams.batch_size, len(self.hparams.out_vars), lat, lon])
        return in_size, out_size

    def get_normalize(self, root_dir, variables):
        """
        Get normalization transforms for the specified variables.
        
        Loads mean and standard deviation values from npz files and creates
        normalization transforms for each variable.
        
        Parameters
        ----------
        root_dir : str
            Directory containing normalization statistics files
        variables : List[str]
            List of variable names to create transforms for
            
        Returns
        -------
        Dict[str, transforms.Normalize]
            Dictionary mapping variable names to normalization transforms
        """
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        return {
            var: transforms.Normalize(normalize_mean[var][0], normalize_std[var][0])
            for var in variables
        }

    def get_out_transforms(self):
        out_transforms = {}
        for key in self.output_transforms.keys():
            out_transforms[key] = self.output_transforms[key]
        return out_transforms

    def get_climatology(self, split):
        path = os.path.join(self.hparams.out_root_dir, split, "climatology.npz")
        clim_dict = np.load(path)
        new_clim_dict = {}
        for var in self.hparams.out_vars:
            new_clim_dict[var] = torch.from_numpy(
                np.squeeze(clim_dict[var].astype(np.float32), axis=0)
            )
        return new_clim_dict

    def setup(self, stage: Optional[str] = None):
        """
        Assign training, validation, and testing datasets.
        
        Creates dataset objects with appropriate configurations for each split:
        - Training: Includes shuffling and data augmentation
        - Validation/Test: No shuffling, used for evaluation
        
        Args:
            stage: Optional stage identifier ("fit" or "test")
        """
        # Common dataset arguments
        base_dataset_args = {
            "variables": self.hparams.in_vars,
            "out_variables": self.hparams.out_vars,
        }
        
        iter_args = {
            "transforms": self.transforms,
            "output_transforms": self.output_transforms, 
            "subsample": self.hparams.subsample
        }

        # Create test dataset if test files exist
        if self.has_test_files:
            test_reader = NpyReader(
                inp_file_list=self.inp_file_list_test,
                out_file_list=self.out_file_list_test,
                shuffle=False,
                **base_dataset_args
            )
            self.data_test = IndividualDataIter(
                self.dataset_caller(test_reader, **self.dataset_arg),
                **iter_args
            )

        # Create training and validation datasets if in training mode
        if stage != "test":
            # Create training dataset with shuffling if train files exist
            if self.has_train_files and not self.data_train:
                train_reader = NpyReader(
                    inp_file_list=self.inp_file_list_train,
                    out_file_list=self.out_file_list_train,
                    shuffle=True,
                    **base_dataset_args
                )
                self.data_train = ShuffleIterableDataset(
                    IndividualDataIter(
                        self.dataset_caller(train_reader, **self.dataset_arg),
                        **iter_args
                    ),
                    buffer_size=self.hparams.buffer_size
                )

            # Create validation dataset if val files exist
            if self.has_val_files and not self.data_val:
                val_reader = NpyReader(
                    inp_file_list=self.inp_file_list_val,
                    out_file_list=self.out_file_list_val,
                    shuffle=False,
                    **base_dataset_args
                )
                self.data_val = IndividualDataIter(
                    self.dataset_caller(val_reader, **self.dataset_arg),
                    **iter_args
                )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if not self.data_val:
            return None
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        if not self.data_test:
            return None
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

# -------------------- helper functions --------------------

def collate_fn(batch):
    """
    Collate function for batching data samples for direct/iterative forecasting tasks.
    
    Handles:
    1. Converting dictionary features to tensors
    2. Stacking batches of samples
    
    Args:
        batch: List of (input, output) tuples
        
    Returns:
        Tuple of (inputs, outputs, variable_names)
    """
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables


def collate_fn_continuous(batch):
    """
    Collate function for batching data samples for continuous forecasting tasks.
    """
    def handle_dict_features(t: Dict[str, torch.tensor]) -> torch.tensor:
        t = torch.stack(tuple(t.values()))
        if len(t.size()) == 4:
            return torch.transpose(t, 0, 1)
        return t

    inp = torch.stack([handle_dict_features(batch[i][0]) for i in range(len(batch))])
    out = torch.stack([handle_dict_features(batch[i][1]) for i in range(len(batch))])
    lead_times = torch.stack([batch[i][2] for i in range(len(batch))])
    b, t, _, h, w = inp.shape
    lead_times = lead_times.reshape(b, 1, 1, 1, 1).repeat(1, t, 1, h, w)
    inp = torch.cat((inp, lead_times), dim=2)
    variables = list(batch[0][0].keys())
    out_variables = list(batch[0][1].keys())
    return inp, out, variables, out_variables
