# Standard library
from typing import Any, Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
import torch

# Local application
from .args import StackedClimateDatasetArgs
from .climate_dataset import ClimateDataset


class StackedClimateDataset(ClimateDataset):
    """
    A wrapper class that combines multiple climate datasets into a unified interface.
    
    This class allows working with multiple climate datasets (e.g., ERA5, CMIP6) as if they
    were a single dataset. It maintains dataset independence by prefixing all variable names
    with their source dataset name while providing a unified access pattern.
    
    - Supports both map-style (all in memory) and shard-style (chunked) loading
    """
    _args_class: Callable[..., StackedClimateDatasetArgs] = StackedClimateDatasetArgs

    def __init__(self, data_args: StackedClimateDatasetArgs) -> None:
        """
        Initialize the stacked dataset with multiple child datasets.
        
        Args:
            data_args: Configuration object containing child dataset arguments
                      and the name for this stacked dataset
        """
        self.climate_datasets: Sequence[ClimateDataset] = []
        # Initialize each child dataset from the provided arguments
        for data_arg in data_args.child_data_args:
            if isinstance(data_arg._data_class, str):
                # Handle string class names by evaluating them
                climate_dataset_class: Callable[..., ClimateDataset] = eval(
                    data_arg._data_class
                )
            else:
                # Use the class directly if already a class object
                climate_dataset_class: Callable[
                    ..., ClimateDataset
                ] = data_arg._data_class
            self.climate_datasets.append(climate_dataset_class(data_arg))
        self.name: str = data_args.name

    def setup(
        self, style: str = "map", setup_args: Dict = {}
    ) -> Tuple[int, Dict[str, Sequence[str]]]:
        """
        Set up all child datasets and ensure they have matching lengths.
        
        Args:
            style: Loading style ("map" or "shard")
            setup_args: Additional setup arguments passed to child datasets
            
        Returns:
            Tuple containing:
            - Length of the datasets (all must be equal)
            - Dictionary mapping prefixed variable names to their dependencies
            
        Raises:
            RuntimeError: If child datasets have different lengths
        """
        dataset_length: Sequence[int] = []
        variables_to_update: Dict[str, Sequence[str]] = {}
        for climate_dataset in self.climate_datasets:
            length, var_to_update = climate_dataset.setup(style, setup_args)
            dataset_length.append(length)
            # Prefix variable names with dataset name for uniqueness
            for key in var_to_update.keys():
                variables_to_update[self.name + ":" + key] = [
                    self.name + ":" + k for k in var_to_update[key]
                ]
        if not len(set(dataset_length)) == 1:
            raise RuntimeError(
                f"Recieved datasets of different lengths: {dataset_length}"
            )
        return dataset_length[0], variables_to_update

    def load_chunk(self, chunk_id: int) -> int:
        """
        Load a specific chunk from all child datasets.
        
        Args:
            chunk_id: Index of the chunk to load
            
        Returns:
            Length of the loaded chunk (all datasets must return same length)
            
        Raises:
            RuntimeError: If chunks from different datasets have different lengths
        """
        dataset_length: Sequence[int] = []
        for climate_dataset in self.climate_datasets:
            length: int = climate_dataset.load_chunk(chunk_id)
            dataset_length.append(length)
        if not len(set(dataset_length)) == 1:
            raise RuntimeError(
                f"Recieved datasets of different lengths: {dataset_length}"
            )
        return dataset_length[0]

    def get_item(self, index: int) -> Dict[str, torch.tensor]:
        """
        Get an item from all child datasets at the specified index.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dictionary mapping prefixed variable names to their tensor values
        """
        item_dict: Dict[str, torch.tensor] = {}
        for dataset in self.climate_datasets:
            child_item_dict: Dict[str, torch.tensor] = dataset.get_item(index)
            for key in child_item_dict.keys():
                item_dict[self.name + ":" + key] = child_item_dict[key]
        return item_dict

    def get_constants_data(self) -> Dict[str, torch.tensor]:
        """
        Get constant field data from all child datasets.
        
        Returns:
            Dictionary mapping prefixed constant names to their tensor values
        """
        constants_data_dict: Dict[str, torch.tensor] = {}
        for dataset in self.climate_datasets:
            child_constants_data_dict: Dict[
                str, torch.tensor
            ] = dataset.get_constants_data()
            for key in child_constants_data_dict.keys():
                constants_data_dict[self.name + ":" + key] = child_constants_data_dict[
                    key
                ]
        return constants_data_dict

    def get_time(self) -> Dict[str, Union[np.ndarray, None]]:
        """
        Get time information from all child datasets.
        
        Returns:
            Dictionary mapping prefixed time variables to their values
        """
        time_dict: Dict[str, Union[np.ndarray, None]] = {}
        for dataset in self.climate_datasets:
            child_time_dict: Dict[str, Union[np.ndarray, None]] = dataset.get_time()
            for key in child_time_dict.keys():
                time_dict[self.name + ":" + key] = child_time_dict[key]
        return time_dict

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from all child datasets.
        
        Returns:
            Dictionary mapping prefixed metadata keys to their values
        """
        metadata_dict: Dict[str, Any] = {}
        for dataset in self.climate_datasets:
            child_metadata_dict: Dict[str, Any] = dataset.get_metadata()
            for key in child_metadata_dict.keys():
                metadata_dict[self.name + ":" + key] = child_metadata_dict[key]
        return metadata_dict


# Link the arguments class to this dataset class
StackedClimateDatasetArgs._data_class = StackedClimateDataset
