# Standard library
import copy
import glob
import os
import random
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

# Third party
import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

# Local application
from ..args import ERA5Args
from ..climate_dataset import ClimateDataset
from .constants import (
    CONSTANTS,
    DEFAULT_PRESSURE_LEVELS,
    NAME_LEVEL_TO_VAR_LEVEL,
    NAME_TO_VAR,
    PRESSURE_LEVEL_VARS,
    SINGLE_LEVEL_VARS,
)


class ERA5(ClimateDataset):
    """
    ERA5 climate dataset implementation.
    
    This class inherits from ClimateDataset to provide specific functionality for working
    with ERA5 reanalysis data.
    """
    _args_class: Callable[..., ERA5Args] = ERA5Args

    def __init__(self, data_args: ERA5Args) -> None:
        """
        Initialize ERA5 dataset with specific configuration.
        
        Extends parent initialization with ERA5-specific attributes:
        - Root directory for data files
        - Years to process
        - Coordinate arrays (lat, lon, time)
        - Variable mapping for pressure levels
        
        Args:
            data_args: ERA5-specific configuration containing root directory and years
        """
        super().__init__(data_args)
        self.root_dir: str = data_args.root_dir
        self.years: Iterable[int] = data_args.years
        # Coordinate arrays initialized as None
        self.lat: Union[np.ndarray, None] = None
        self.lon: Union[np.ndarray, None] = None
        self.time: Union[np.ndarray, None] = None
        # Maps base variable names to their full names (including pressure levels)
        self.variables_map: Dict[str, Sequence[str]] = {}
        self.build_variables_map()
        # Storage for constant fields and variable data
        self.constants_data: Dict[str, torch.tensor] = {}
        self.data_dict: Dict[
            str, Union[torch.tensor, Sequence]
        ] = self.initialize_data_dict()

    def build_variables_map(self) -> None:
        for name in self.variables:
            # Single-level surface variables
            if name in SINGLE_LEVEL_VARS:
                self.variables_map[name] = [name]
            # Specific pressure level of a variable
            elif name in NAME_LEVEL_TO_VAR_LEVEL:
                self.variables_map[name] = [name]
            # Variable at all pressure levels
            elif name in PRESSURE_LEVEL_VARS:
                self.variables_map[name] = []
                for level in DEFAULT_PRESSURE_LEVELS:
                    self.variables_map[name].append(f"{name}_{level}")
            else:
                raise NotImplementedError(
                    f"{name} is not either in single-level or pressure-level dict"
                )

    def variables_to_update_for_task(self) -> Dict[str, Sequence[str]]:
        return self.variables_map

    def get_file_name_from_variable(self, var: str) -> str:
        if var in SINGLE_LEVEL_VARS or var in PRESSURE_LEVEL_VARS:
            return var
        else:
            return "_".join(var.split("_")[:-1])

    def set_lat_lon(self, year: int) -> None:
        """
        Extract latitude and longitude information from a data file.
        
        Loads coordinate information from the first variable's file for a given year.
        This information is the same across all variables, so we only need to load it once.
        
        Args:
            year: Year from which to extract coordinate information
            
        Raises:
            RuntimeError: If anything other than one matching file is found
        """
        dir_var = os.path.join(
            self.root_dir, self.get_file_name_from_variable(self.variables[0])
        )
        ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
        if len(ps) != 1:
            raise RuntimeError(
                f"Found {len(ps)} files corresponding to the {self.variables[0]}"
                f" for the year {year}."
            )
        xr_data = xr.load_dataset(ps[0])
        self.lat = xr_data["lat"].values
        self.lon = xr_data["lon"].values

    def setup_metadata(self, year: int) -> None:
        ## Prevent setup if already called before
        if self.lat is None or self.lon is None:
            self.set_lat_lon(year)

    def setup_constants(self, data_dir: str) -> None:
        """
        Load constant fields (e.g., land-sea mask, orography).
        
        Implements the abstract method from ClimateDataset.
        Loads constant fields from a single NetCDF file and converts them to tensors.
        Only loads if not already loaded.
        
        Args:
            data_dir: Directory containing the constants file
            
        Raises:
            RuntimeError: If exactly one constants file is not found
        """
        if self.constants_data != {}:
            return
        if len(self.constants) > 0:
            ps = glob.glob(os.path.join(data_dir, "constants", "*.nc"))
            if len(ps) != 1:
                raise RuntimeError(
                    f"Found {len(ps)} files corresponding to the constants ."
                    f"Should be just one."
                )
            all_constants = xr.load_dataset(ps[0])
            for name in self.constants:
                self.constants_data[name] = torch.tensor(
                    all_constants[NAME_TO_VAR[name]].values.astype(np.float32)
                )

    def initialize_data_dict(self) -> Dict[str, Sequence]:
        """
        Initialize empty dictionary for storing variable data.
        
        Creates entries for all variables, including pressure-level variants.
        
        Returns:
            Empty dictionary with keys for all variables
        """
        data_dict: Dict[str, Sequence] = {}
        for variables in self.variables_map.values():
            for variable in variables:
                data_dict[variable] = []
        return data_dict

    def load_from_nc_by_years(self, data_dir: str, years) -> Dict[str, torch.tensor]:
        """
        Load data for specified years from NetCDF files.
        
        This method:
        1. Loads data for each variable and year
        2. Handles both single-level and pressure-level variables
        3. Concatenates data across years
        4. Converts to PyTorch tensors
        
        Args:
            data_dir: Root directory containing variable subdirectories
            years: Sequence of years to load
            
        Returns:
            Dictionary mapping variable names to their data tensors
        """
        data_dict: Dict[
            str, Sequence[xr.core.dataarray.DataArray]
        ] = self.initialize_data_dict()
        for year in tqdm(years):
            for var in self.variables:
                dir_var = os.path.join(data_dir, self.get_file_name_from_variable(var))
                ps = glob.glob(os.path.join(dir_var, f"*{year}*.nc"))
                if len(ps) != 1:
                    raise RuntimeError(
                        f"Found {len(ps)} files corresponding to the {var} for the year {year}."
                    )
                xr_data = xr.load_dataset(ps[0])
                xr_data = xr_data[NAME_TO_VAR[self.get_file_name_from_variable(var)]]
                # np_data = xr_data.to_numpy()
                if len(xr_data.shape) == 3:  # 8760, 32, 64
                    data_dict[var].append(xr_data)
                else:  # multi level data
                    if var in NAME_LEVEL_TO_VAR_LEVEL:  ## loading only a specific level
                        level = int(var.split("_")[-1])
                        xr_data_level = (xr_data.sel(level=[level])).squeeze(axis=1)
                        data_dict[var].append(xr_data_level)
                    else:  ## loading all levels
                        for level in DEFAULT_PRESSURE_LEVELS:
                            xr_data_level = (xr_data.sel(level=[level])).squeeze(axis=1)
                            data_dict[f"{var}_{level}"].append(xr_data_level)

        data_dict: Dict[str, xr.core.dataarray.DataArray] = {
            k: xr.concat(data_dict[k], dim="time") for k in data_dict.keys()
        }
        # precipitation and solar radiation miss a few data points in the beginning
        len_min = min([data_dict[k].shape[0] for k in data_dict.keys()])
        data_dict = {k: data_dict[k][-len_min:] for k in data_dict.keys()}
        # using next(iter) isntead of list(data_dict.keys())[0] to get random element
        self.time = data_dict[next(iter(data_dict.keys()))].time.values
        data_dict: Dict[str, torch.tensor] = {
            k: torch.from_numpy(data_dict[k].values.astype(np.float32))
            for k in data_dict.keys()
        }
        return data_dict

    def setup_map(self) -> Tuple[int, Dict[str, Sequence[str]]]:
        """
        Set up dataset in "map" mode - loading all data into memory at once.
        
        Returns:
            Tuple containing:
            - Length of dataset (number of time steps)
            - Dictionary mapping variables to their dependencies
        """
        self.setup_constants(self.root_dir)
        self.setup_metadata(self.years[0])
        self.data_dict = self.load_from_nc_by_years(self.root_dir, self.years)
        return (
            len(self.data_dict[next(iter(self.data_dict.keys()))]),
            self.variables_to_update_for_task(),
        )

    def build_years_to_iterate(
        self, seed: int, drop_last: bool, shuffle: bool
    ) -> Sequence[int]:
        """
        Create sequence of years for distributed data loading.
        
        Handles distributed training scenarios by:
        1. Optionally shuffling years
        2. Dividing years among workers
        3. Managing uneven divisions with drop_last option
        
        Args:
            seed: Random seed for shuffling
            drop_last: Whether to drop remainder years that don't divide evenly
            shuffle: Whether to shuffle years before dividing
            
        Returns:
            Sequence of years assigned to this worker
        """
        temp_years = list(copy.deepcopy(self.years))
        if shuffle:
            random.Random(seed).shuffle(temp_years)
        if drop_last:
            n_years = len(temp_years) // self.world_size
        else:
            n_years = (len(temp_years) + self.world_size - 1) // self.world_size
        years_to_iterate = []
        start_index = self.rank * n_years
        for _ in range(n_years):
            years_to_iterate.append(temp_years[start_index])
            start_index = (start_index + 1) % len(temp_years)
        return years_to_iterate

    def setup_shard(self, args: Dict[str, int]) -> Tuple[int, Dict[str, Sequence[str]]]:
        """
        Set up dataset in "shard" mode for distributed loading.
        
        Implements the shard-style loading from ClimateDataset:
        1. Validates required arguments for distributed setup
        2. Assigns years to workers
        3. Initializes data structures for chunk loading
        
        Args:
            args: Dictionary containing:
                - world_size: Total number of workers
                - rank: This worker's rank
                - seed: Random seed for shuffling
                - n_chunks: Number of chunks to divide data into
                - drop_last: (optional) Whether to drop remainder data
                - shuffle: (optional) Whether to shuffle data
                
        Returns:
            Tuple containing:
            - -1 (placeholder length, actual length determined per chunk)
            - Dictionary mapping variables to their dependencies
            
        Raises:
            RuntimeError: If required arguments are missing or invalid
        """
        if not "world_size" in args.keys():
            raise RuntimeError(f"Required world_size as key in the setup_shard args")
        if not "rank" in args.keys():
            raise RuntimeError(f"Required rank as key in the setup_shard args")
        if not "seed" in args.keys():
            raise RuntimeError(f"Required seed as key in the setup_shard args")
        if not "n_chunks" in args.keys():
            raise RuntimeError(f"Required n_chunks as key in the setup_shard args")

        self.world_size: int = args["world_size"]
        self.rank: int = args["rank"]
        self.n_chunks: int = args["n_chunks"]

        if "drop_last" in args.keys():
            drop_last = True
        else:
            drop_last = False

        if "shuffle" in args.keys():
            shuffle = True
        else:
            shuffle = False

        years_to_iterate: Sequence[int] = self.build_years_to_iterate(
            args["seed"], drop_last, shuffle
        )
        self.setup_constants(self.root_dir)
        self.setup_metadata(years_to_iterate[0])

        if len(years_to_iterate) < self.n_chunks:
            RuntimeError(
                f"Number of chunks:{self.n_chunks} are more than "
                f"the available years: {len(years_to_iterate)}"
            )
        self.years_to_iterate: Sequence[int] = years_to_iterate
        self.chunks_iterated_till_now: int = 0
        self.data_dict = self.initialize_data_dict()
        return -1, self.variables_to_update_for_task()

    def load_chunk(self, chunk_id: int) -> int:
        """
        Load a specific chunk of data in shard mode.
        Args:
            chunk_id: Index of the chunk to load
            
        Returns:
            Number of time steps in the loaded chunk
        """
        n_years_in_chunk: int = len(self.years_to_iterate) // self.n_chunks
        n_chunks_with_extra_year: int = len(self.years_to_iterate) % self.n_chunks
        offset: int = 0
        if chunk_id >= n_chunks_with_extra_year:
            offset = n_chunks_with_extra_year * (n_years_in_chunk + 1)
            chunk_id = chunk_id - n_chunks_with_extra_year
        else:
            n_years_in_chunk = n_years_in_chunk + 1
        years_to_iterate_this_chunk = self.years_to_iterate[
            offset
            + chunk_id * n_years_in_chunk : offset
            + (chunk_id + 1) * n_years_in_chunk
        ]
        self.data_dict = self.load_from_nc_by_years(
            self.root_dir, years_to_iterate_this_chunk
        )
        return len(self.data_dict[next(iter(self.data_dict.keys()))])

    def setup(
        self, style: str = "map", setup_args: Dict = {}
    ) -> Tuple[int, Dict[str, Sequence[str]]]:
        """
        Main setup method implementing ClimateDataset interface.
        
        Delegates to appropriate setup method based on style:
        - "map": Load all data at once
        - "shard": Load data in chunks
        
        Also handles variable name prefixing for dataset identification.
        
        Args:
            style: Loading style ("map" or "shard")
            setup_args: Additional arguments for setup
            
        Returns:
            Tuple containing:
            - Dataset length
            - Dictionary mapping prefixed variable names to their dependencies
        """
        supported_styles: Sequence[str] = ["map", "shard"]
        if style == "map":
            length, var_to_update = self.setup_map()
        elif style == "shard":
            length, var_to_update = self.setup_shard(setup_args)
        else:
            RuntimeError(
                f"Please choose a valid style of loading data. "
                f"Current available options include: {supported_styles}. "
                f"You have choosen: {style}"
            )
        variables_to_update: Dict[str, Sequence[str]] = {}
        for var in var_to_update.keys():
            variables_to_update[self.name + ":" + var] = [
                self.name + ":" + v for v in var_to_update[var]
            ]
        return length, variables_to_update

    def get_item(
        self, index: int
    ) -> Dict[str, torch.tensor]:  # Dict where each value is a torch tensor shape 32*64
        return {
            self.name + ":" + k: self.data_dict[k][index] for k in self.data_dict.keys()
        }

    def get_constants_data(
        self,
    ) -> Dict[str, torch.tensor]:  # Dict where each value is a torch tensor shape 32*64
        return {
            self.name + ":" + k: self.constants_data[k]
            for k in self.constants_data.keys()
        }

    def get_time(self) -> Dict[str, Union[np.ndarray, None]]:
        return {self.name + ":time": self.time}

    def get_metadata(
        self,
    ) -> Dict[str, Union[np.ndarray, None]]:  # Dict where each value is a ndarray
        return {self.name + ":lat": self.lat, self.name + ":lon": self.lon}


ERA5Args._data_class = ERA5
