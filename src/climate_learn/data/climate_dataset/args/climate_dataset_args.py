# Standard library
from __future__ import annotations  # Enable forward references in type hints
from abc import ABC
import copy
from typing import Any, Callable, Dict, Sequence, TYPE_CHECKING, Union

# Local application
if TYPE_CHECKING:
    from ..climate_dataset import ClimateDataset


class ClimateDatasetArgs(ABC):
    """
    Base arguments class for climate datasets.
    
    This abstract class serves as a configuration container for climate datasets,
    storing and validating the basic parameters needed to initialize a dataset.
    It provides a standardized way to:
    1. Specify which climate variables and constants to load
    2. Create modified copies of configurations
    3. Validate configuration parameters
    
    The class is designed to be subclassed for specific dataset types (e.g., ERA5Args,
    CMIP6Args) that may require additional parameters.
    """
    # Reference to the dataset class this arguments class configures
    # Can be either a class reference or string (for lazy loading)
    _data_class: Union[Callable[..., ClimateDataset], str] = "ClimateDataset"

    def __init__(
        self,
        variables: Sequence[str],
        constants: Sequence[str] = [],
        name: str = "climate_dataset",
    ) -> None:
        """
        Initialize dataset arguments.

        Args:
            variables: List of climate variables to load (e.g., ["temperature", "precipitation"])
                      These are the main time-varying fields the dataset will provide.
            
            constants: List of constant fields to load (e.g., ["land_sea_mask", "orography"])
                      These are time-invariant fields that provide context for the variables.
                      Defaults to empty list.
            
            name: Identifier for this dataset configuration.
                 Used especially in stacked datasets to prefix variable names.
                 Defaults to "climate_dataset".

        Example:
            >>> args = ClimateDatasetArgs(
            ...     variables=["2m_temperature", "total_precipitation"],
            ...     constants=["land_sea_mask"],
            ...     name="era5_dataset"
            ... )
        """
        self.variables: Sequence[str] = variables
        self.constants: Sequence[str] = constants
        self.name: str = name
        ClimateDatasetArgs.check_validity(self)

    def create_copy(self, args: Dict[str, Any] = {}) -> ClimateDatasetArgs:
        """
        Create a modified copy of this configuration.
        
        This method is useful when you need a similar configuration with just a few
        changes, without modifying the original configuration.

        Args:
            args: Dictionary of attributes to modify in the copy.
                 Keys are attribute names, values are the new values.

        Returns:
            A new ClimateDatasetArgs instance with the specified modifications.

        Example:
            >>> original_args = ClimateDatasetArgs(variables=["temperature"])
            >>> modified_args = original_args.create_copy({"name": "new_dataset"})
        """
        new_instance: ClimateDatasetArgs = copy.deepcopy(self)
        for arg in args:
            if hasattr(new_instance, arg):
                setattr(new_instance, arg, args[arg])
        ClimateDatasetArgs.check_validity(new_instance)
        return new_instance

    def check_validity(self) -> None:
        """
        Validate the configuration parameters.
        
        This method should be overridden by subclasses to implement specific validation
        rules. 
        
        Raises:
            RuntimeError: If validation fails (in subclass implementations)
        """
        pass
