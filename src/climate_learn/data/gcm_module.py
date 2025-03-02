"""
This simple module connects the GCM datasets from processing.py with the IterDataModule.
It provides functionality to create data modules for ExoCAM, LMDG, ROCKE3D, and UM datasets
that can be used with the model loading utilities in climate_learn.utils.loaders.
"""

# Standard library
import os
import glob
from typing import List, Optional, Union, Dict, Any

# Third party
import numpy as np
import torch
import pytorch_lightning as pl

# Local application
from .itermodule import IterDataModule
from .utils import create_data_splits
from .processing.processing import (
    GCMDataset,
    ExoCAMDataset,
    LMDGDataset,
    ROCKE3DDataset,
    UMDataset
)

class GCMDataModule(IterDataModule):
    """
    PyTorch Lightning DataModule for GCM datasets.
    
    This class extends IterDataModule to work specifically with GCM datasets
    processed using the GCMDataset classes from processing.py. It assumes
    data has already been processed into the standard format expected by
    IterDataModule (with train/val/test splits and climatology files).
    
    Attributes:
        Same as IterDataModule, plus:
        gcm (str): Type of GCM dataset ("exocam", "lmdg", "rocke3d", "um")
    """
    
    def __init__(
        self,
        task: str,
        in_vars: List[str],
        out_vars: List[str],
        data_dir: str,
        splits: List[float] = None,
        src: str = None,
        vars_3d_to_2d: bool = True,
        history: int = 1,
        window: int = 6,
        pred_range: int = 6,
        random_lead_time: bool = True, # continuous forecasting only
        max_pred_range: int = 120, # continuous forecasting only
        hrs_each_step: int = 1, # continuous forecasting only
        subsample: int = 1,
        buffer_size: int = 10000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Initialize a GCMDataModule.
        
        Parameters
        ----------
        task : str
            Type of prediction task 
            ("direct-forecasting", "iterative-forecasting", 
             "continuous-forecasting", "downscaling")
        in_vars : List[str]
            List of input variable names to use
        out_vars : List[str]
            List of output variable names to predict
        data_dir : str
            Directory containing the processed GCM data (train/val/test subfolders etc)
        splits : List[float]
            Proportions for data splits. Length determines number of splits:
            - Length 2: [train, val] proportions
            - Length 3: [train, val, test] proportions
            Values should sum to 1.0
            If None, defaults to all in train set.
        vars_3d_to_2d : bool
            Whether to automatically expand 3D variables into their 2D level-specific versions
        history : int
            Number of historical time steps to use
        window : int
            Window size for temporal processing
        pred_range : int
            Prediction horizon (in time steps)
        random_lead_time : bool
            Whether to use random lead times
        max_pred_range : int
            Maximum prediction horizon
        hrs_each_step : int
            Hours between time steps
        subsample : int
            Subsampling factor for data
        buffer_size : int
            Size of shuffle buffer
        batch_size : int
            Number of samples per batch
        num_workers : int
            Number of data loading workers
        pin_memory : bool
            Whether to pin memory in data loading
        """
        # If vars_3d_to_2d is True, expand 3D variable names into their 2D level-specific versions
        if vars_3d_to_2d:
            in_vars, out_vars = self._expand_3d_variables(data_dir, in_vars, out_vars)

        # Create train/val/test splits
        if splits is not None and splits != []: # if none or empty list, all data is in train set
            create_data_splits(
                data_dir, 
                splits=splits
            )
        
        super().__init__(
            task=task,
            inp_root_dir=data_dir, # data_dir is both input and output root directory for GCM data
            out_root_dir=data_dir,
            in_vars=in_vars,
            out_vars=out_vars,
            src=src,
            history=history,
            window=window,
            pred_range=pred_range,
            random_lead_time=random_lead_time,
            max_pred_range=max_pred_range,
            hrs_each_step=hrs_each_step,
            subsample=subsample,
            buffer_size=buffer_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _expand_3d_variables(self, data_dir: str, in_vars: List[str], out_vars: List[str]) -> tuple:
        """
        Expand 3D variables into their 2D level-specific versions.
        
        This method scans the processed data directory to find level-specific versions
        of 3D variables (e.g., temperature_0, temperature_1, etc.) and replaces the
        original variable names with these expanded versions.
        
        Parameters
        ----------
        data_dir : str
            Directory containing the processed data
        in_vars : List[str]
            List of input variable names
        out_vars : List[str]
            List of output variable names
            
        Returns
        -------
        tuple
            Tuple of (expanded_in_vars, expanded_out_vars)
        """
        train_dir = os.path.join(data_dir, "train")
            
        # Find all NPZ files in the train directory (excluding climatology)
        npz_files = [f for f in os.listdir(train_dir) 
                    if f.endswith(".npz") and not f.startswith("climatology")]
        
        sample_file = os.path.join(train_dir, npz_files[0])
        expanded_vars = {}
        with np.load(sample_file) as data:
            # Get all keys from the NPZ file
            all_keys = list(data.keys())
            
            # Check each variable for level-specific versions
            all_vars = set(in_vars) | set(out_vars)
            for var in all_vars:
                # Find all keys that match the pattern var_N where N is a number
                level_vars = [k for k in all_keys if k.startswith(f"{var}_") and 
                                k[len(var)+1:].isdigit()]
                
                if level_vars:
                    # Sort by level index
                    level_vars.sort(key=lambda x: int(x[len(var)+1:]))
                    expanded_vars[var] = level_vars
        
        # Expand the input and output variable lists
        expanded_in_vars = []
        for var in in_vars:
            if var in expanded_vars:
                expanded_in_vars.extend(expanded_vars[var])
            else:
                expanded_in_vars.append(var)
                
        expanded_out_vars = []
        for var in out_vars:
            if var in expanded_vars:
                expanded_out_vars.extend(expanded_vars[var])
            else:
                expanded_out_vars.append(var)
                
        return expanded_in_vars, expanded_out_vars

    @classmethod
    def create_from_netcdf(
        cls,
        gcm: str,
        gcm_kwargs: Dict[str, Any],
        task: str,
        nc_files: Union[str, List[str]],
        data_dir: str,
        variables_to_extract: List[str],
        lat_target: np.ndarray,
        lon_target: np.ndarray,
        P_levels: np.ndarray,
        shard_size: Optional[int] = None,
        all_2d: bool = True,
        in_vars: List[str] = None,
        out_vars: List[str] = None,
        datamodule_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a GCMDataModule by directly processing NetCDF files.
        
        This method:
        1. Processes NetCDF files using the appropriate GCMDataset class
        2. Returns a configured GCMDataModule ready for use
        
        Parameters
        ----------
        gcm : str
            which GCM dataset ("exocam", "lmdg", "rocke3d", "um")
        gcm_kwargs : Optional[Dict[str, Any]]
            Additional arguments to pass to the GCM dataset constructor (lon_substellar + g, R_d, R_v for exocam)
        task : str
            Type of prediction task
        nc_files : Union[str, List[str]]
            Path(s) to NetCDF file(s) to process
        data_dir : str
            Directory to save processed data
        in_vars : List[str]
            List of input variable names to use
        out_vars : List[str]
            List of output variable names to predict
        variables_to_extract : List[str]
            Variables to extract from the NetCDF files
        lat_target : np.ndarray
            Target latitude grid
        lon_target : np.ndarray
            Target longitude grid
        P_levels : np.ndarray
            Target pressure levels
        shard_size : Optional[int]
            Size of data shards (if None, all data in one file)
        all_2d : bool
            Whether to split 3D variables into 2D variables
        datamodule_kwargs : Dict[str, Any]
            Additional arguments to pass to GCMDataModule constructor, from list below:
                src : str
                splits : List[float]
                vars_3d_to_2d : bool
                history : int
                window : int
                pred_range : int
                random_lead_time : bool
                max_pred_range : int
                hrs_each_step : int
                subsample : int
                buffer_size : int
                batch_size : int
                num_workers : int
                pin_memory : bool
                
        Returns
        -------
        GCMDataModule
            Configured data module ready for use
        """
        gcm = gcm.lower()
        datamodule_kwargs = datamodule_kwargs or {}
        
        # Process NetCDF files using the appropriate GCMDataset class
        if gcm == "exocam":
            dataset = ExoCAMDataset(nc_files, **gcm_kwargs)
        elif gcm == "lmdg":
            dataset = LMDGDataset(nc_files, **gcm_kwargs)
        elif gcm == "rocke3d":
            dataset = ROCKE3DDataset(nc_files, **gcm_kwargs)
        elif gcm == "um":
            dataset = UMDataset(nc_files, **gcm_kwargs)
        else:
            raise ValueError(f"Unsupported GCM type: {gcm}")
        
        # Process the data and save to data_dir
        dataset.to_npz(
            save_dir=data_dir,
            variables=variables_to_extract,
            lat_target=lat_target,
            lon_target=lon_target,
            P_levels=P_levels,
            shard_size=shard_size, 
            all_2d=all_2d
        )

        # Convert processed data into DataModule
        
        
        # Create and return the data module
        return cls(
            gcm=gcm,
            task=task,
            in_vars=in_vars,
            out_vars=out_vars,
            data_dir=data_dir,
            **datamodule_kwargs
        )

