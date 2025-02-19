# Standard library
from abc import ABC # Abstract Base Class
from typing import Any, Callable, Dict, Sequence, Tuple, Union

# Third party
import numpy as np
import torch

# Local application
from .args import ClimateDatasetArgs


class ClimateDataset(ABC):
    """
    Abstract base class for climate datasets.
    
    This class defines the interface for working with climate datasets, providing methods
    for loading, accessing, and managing climate data and metadata. Concrete implementations
    should inherit from this class and implement the abstract methods.
    
    The class supports two main data loading styles:
    - "map": Loads all data into memory at once
    - "shard": Loads data in chunks/shards for memory efficiency
    """
    # Class that handles initialization arguments for this dataset
    _args_class: Callable[..., ClimateDatasetArgs] = ClimateDatasetArgs

    def __init__(self, data_args: ClimateDatasetArgs) -> None:
        """
        Initialize the climate dataset.

        Args:
            data_args: Configuration object containing dataset parameters like variables to load
        """
        self.variables: Sequence[str] = data_args.variables  # Climate variables to load (e.g. temperature, pressure)
        self.constants: Sequence[str] = data_args.constants  # Constant fields (e.g. land-sea mask, orography)
        self.name: str = data_args.name  # Identifier for this dataset

    def setup_constants(self) -> None:
        """Load constant fields that don't vary with time (e.g. land-sea mask, orography)"""
        raise NotImplementedError

    def setup_metadata(self) -> None:
        """Load dataset metadata (e.g. lat/lon coordinates, time information)"""
        raise NotImplementedError

    def setup_map(self) -> Tuple[int, Any]:
        """
        Setup dataset in "map" mode - load all data into memory at once.
        
        Returns:
            Tuple containing:
            - Length of dataset (number of samples)
            - Dictionary mapping variable names to their dependencies
        """
        self.setup_constants()
        self.setup_metadata()
        return -1, {}

    def setup_shard(self, setup_args: dict = {}) -> Tuple[int, Any]:
        """
        Setup dataset in "shard" mode - prepare for loading data in chunks.
        
        Args:
            setup_args: Additional arguments for shard setup (e.g. shard size)
            
        Returns:
            Same as setup_map()
        """
        self.setup_constants()
        self.setup_metadata()
        return -1, {}

    def setup(
        self, style: str = "map", setup_args: Dict[str, Any] = {}
    ) -> Tuple[int, Any]:
        """
        Main setup method that initializes the dataset for use.
        
        Args:
            style: Data loading style - either "map" or "shard"
            setup_args: Additional setup arguments
            
        Returns:
            Tuple containing:
            - Dataset length
            - Dictionary mapping fully qualified variable names (dataset:variable) 
              to their dependencies
        
        Raises:
            RuntimeError: If an invalid style is specified
        """
        supported_styles: Sequence[str] = ["map", "shard"]
        if style == "map":
            length, var_to_update = self.setup_map()
        elif style == "shard":
            length, var_to_update = self.setup_shard(setup_args)
        else:
            raise RuntimeError(
                f"Please choose a valid style of loading data. "
                f"Current available options include: {supported_styles}. "
                f"You have choosen: {style}"
            )
        
        # Prefix variable names with dataset name for uniqueness
        variables_to_update: Dict[str, Sequence[str]] = {}
        for var in var_to_update.keys():
            variables_to_update[self.name + ":" + var] = [
                self.name + ":" + v for v in var_to_update[var]
            ]
        return length, variables_to_update

    def load_chunk(self, chunk_id: int) -> int:
        """
        Load a specific chunk/shard of data when using shard mode.
        
        Args:
            chunk_id: Index of the chunk to load
            
        Returns:
            Length of the loaded chunk
        """
        raise NotImplementedError

    def get_item(self, index: int) -> Dict[str, torch.tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary mapping variable names to their tensor values
        """
        raise NotImplementedError

    def get_constants_data(self) -> Dict[str, torch.tensor]:
        """
        Get the constant field data.
        
        Returns:
            Dictionary mapping constant names to their tensor values
        """
        raise NotImplementedError

    def get_time(self) -> Dict[str, Union[np.ndarray, None]]:
        """
        Get time coordinate information.
        
        Returns:
            Dictionary containing time values/coordinates
        """
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing metadata like lat/lon coordinates
        """
        raise NotImplementedError


# Set the data class reference in the arguments class
ClimateDatasetArgs._data_class = ClimateDataset
