# Standard library
from __future__ import annotations
from typing import Callable, Iterable, Sequence, TYPE_CHECKING, Union

# Local application
from .climate_dataset_args import ClimateDatasetArgs

if TYPE_CHECKING:
    from ..era5.era5 import ERA5


class ERA5Args(ClimateDatasetArgs):
    """Arguments class for ERA5 climate datasets."""
    _data_class: Union[Callable[..., ERA5], str] = "ERA5"

    def __init__(
        self,
        root_dir: str,  # Directory containing ERA5 data files
        variables: Sequence[str], 
        years: Iterable[int],  
        constants: Sequence[str] = [],  # Constant fields like land-sea mask
        name: str = "era5",  # Identifier for this dataset configuration
    ) -> None:
        super().__init__(variables, constants, name)
        self.root_dir: str = root_dir
        self.years: Iterable[int] = years
