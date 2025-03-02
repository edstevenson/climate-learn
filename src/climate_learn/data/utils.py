"""
Utility functions for data processing and manipulation in climate-learn.

This module provides standalone utility functions for working with climate data,
including functions for creating train/val/test splits from processed data.
"""

# Standard library
import os
import glob
from typing import List, Optional

# Third party
import numpy as np


def create_data_splits(
    data_dir: str, 
    splits: List[float]
) -> None:
    """
    Create train/val/test splits from processed data based on provided proportions.
    
    This function takes a directory containing processed climate data (in .npz format)
    and creates train/val/test splits according to the specified proportions.
    It assumes all data is initially in a 'train' subdirectory and will:
    1. Move files to 'val' and 'test' directories according to split proportions
    2. Calculate and save climatology files for each split
    
    Parameters
    ----------
    data_dir : str
        Directory containing processed data with a 'train' subdirectory
    splits : List[float]
        Proportions for data splits. Length determines number of splits:
        - Length 2: [train, val] proportions
        - Length 3: [train, val, test] proportions
        Values should sum to 1.0
    """
    # Validate splits parameter
    if len(splits) not in [2, 3]:
        raise ValueError(f"splits must have length 2 or 3, got {len(splits)}")
    
    if abs(sum(splits) - 1.0) > 1e-10:
        raise ValueError(f"splits proportions must sum to 1, got {splits}")
    
    include_test_set = (len(splits) == 3)
    
    # Create val and test directories
    val_dir = os.path.join(data_dir, "val")
    os.makedirs(val_dir, exist_ok=False)
    if include_test_set:
        test_dir = os.path.join(data_dir, "test")
        os.makedirs(test_dir, exist_ok=False)
    
    # Get all data files in train directory
    train_dir = os.path.join(data_dir, "train")
    train_clim_file = os.path.join(train_dir, "climatology.npz")
    if not os.path.exists(train_clim_file):
        raise FileNotFoundError(f"Climatology file not found in {train_dir}")
    
    # Create train/val/test splits from train data
    data_files = sorted([f for f in glob.glob(os.path.join(train_dir, "*.npz")) 
                        if "climatology" not in f])
    
    n_files = len(data_files)
    n_train = int(n_files * splits[0])
    n_val = int(n_files * splits[1])

    print(f"Total files: {n_files}")
    print(f"Train files: {n_train}")
    print(f"Validation files: {n_val}")
    if include_test_set:
        n_test = n_files - n_train - n_val
        print(f"Test files: {n_test}")
    
    # Dictionary to store data for each splits
    train_data = {}
    val_data = {}
    test_data = {}
    
    # Process each file according to its splits assignment
    for i, file_path in enumerate(data_files):
        with np.load(file_path) as data:
            file_data = dict(data)
            
            if i < n_train:
                # Train set - keep in place but collect data for climatology
                for var, arr in file_data.items():
                    if var not in train_data:
                        train_data[var] = []
                    train_data[var].append(arr)
            elif i < n_train + n_val:
                # Validation set
                for var, arr in file_data.items():
                    if var not in val_data:
                        val_data[var] = []
                    val_data[var].append(arr)
                # Save to validation directory
                np.savez(os.path.join(val_dir, os.path.basename(file_path)), **file_data)
                # Remove from train directory
                os.remove(file_path)
            elif include_test_set:
                # Test set
                for var, arr in file_data.items():
                    if var not in test_data:
                        test_data[var] = []
                    test_data[var].append(arr)
                # Save to test directory
                np.savez(os.path.join(test_dir, os.path.basename(file_path)), **file_data)
                # Remove from train directory
                os.remove(file_path)
    
    # Calculate and save climatology for each splits
    
    # Train climatology
    if train_data:
        train_climatology = {}
        for var, arrays in train_data.items():
            # Concatenate all arrays along the time dimension (axis 0)
            combined = np.concatenate(arrays, axis=0)
            # Calculate mean along time dimension
            train_climatology[var] = combined.mean(axis=0)
        np.savez(os.path.join(train_dir, "climatology.npz"), **train_climatology)
    
    # Validation climatology
    if val_data:
        val_climatology = {}
        for var, arrays in val_data.items():
            combined = np.concatenate(arrays, axis=0)
            val_climatology[var] = combined.mean(axis=0)
        np.savez(os.path.join(val_dir, "climatology.npz"), **val_climatology)
    else:
        # If no validation data, copy train climatology
        with np.load(train_clim_file) as data:
            np.savez(os.path.join(val_dir, "climatology.npz"), **dict(data))
    
    # Test climatology (if needed)
    if include_test_set:
        if test_data:
            test_climatology = {}
            for var, arrays in test_data.items():
                combined = np.concatenate(arrays, axis=0)
                test_climatology[var] = combined.mean(axis=0)
            np.savez(os.path.join(test_dir, "climatology.npz"), **test_climatology)
        else:
            # If no test data, copy train climatology
            with np.load(train_clim_file) as data:
                np.savez(os.path.join(test_dir, "climatology.npz"), **dict(data)) 