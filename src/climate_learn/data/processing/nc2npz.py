# Standard library
import glob
import os

# Third party
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm

# Local application
from .era5_constants import (
    DEFAULT_PRESSURE_LEVELS,
    NAME_TO_VAR,  # Dictionary mapping more readable names to ERA5 variable codes
    VAR_TO_NAME,  # inverse of NAME_TO_VAR
    CONSTANTS,    # List of constant fields (e.g., land-sea mask, orography)
)

# Number of hours per year, adjusted to be divisible by 16 for efficient processing
HOURS_PER_YEAR = 8736  # 8760 --> 8736 which is dividable by 16


def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    """
    Convert NetCDF (.nc) climate data files to NumPy (.npz) format with preprocessing.
    
    This function:
    1. Loads climate data from NetCDF files
    2. Processes both constant fields (e.g., land-sea mask) and time-varying fields
    3. Computes normalization statistics for training data
    4. Handles both surface-level and pressure-level variables
    5. Saves data in sharded format for efficient loading
    
    Args:
        path: Root directory containing NetCDF files
        variables: List of climate variables to process
        years: List of years to process
        save_dir: Directory to save processed .npz files
        partition: Data partition ('train', 'val', or 'test')
        num_shards_per_year: Number of shards to split each year's data into
    """
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    # Initialize dictionaries for normalization statistics (only for training data)
    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}  # Store climatological means

    constants_path = os.path.join(path, "constants.nc")
    constants_are_downloaded = os.path.isfile(constants_path)

    if constants_are_downloaded:
        # Load and process constant fields
        constants = xr.open_mfdataset(
            constants_path, combine="by_coords", parallel=True
        )
        constant_fields = [VAR_TO_NAME[v] for v in CONSTANTS if v in VAR_TO_NAME.keys()]
        constant_values = {}
        for f in constant_fields:
            # Expand constant fields to match temporal dimension
            constant_values[f] = np.expand_dims(
                constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)
            ).repeat(HOURS_PER_YEAR, axis=0)
            if partition == "train":
                # Compute normalization statistics for constant fields
                normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
                normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    # Process each year's data
    for year in tqdm(years):
        np_vars = {}

        # Add constant fields if available
        if constants_are_downloaded:
            for f in constant_fields:
                np_vars[f] = constant_values[f]

        # Process time-varying fields
        for var in variables:
            # Load all NetCDF files for this variable and year
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(
                ps, combine="by_coords", parallel=True
            )
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # Surface-level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                
                # Special processing for precipitation data: remove last 24 hours if this year has 366 days
                if code == "tp":  # Total precipitation
                    tp = ds[code].to_numpy()
                    # Compute 6-hour accumulated precipitation
                    tp_cum_6hrs = np.cumsum(tp, axis=0)
                    tp_cum_6hrs[6:] = tp_cum_6hrs[6:] - tp_cum_6hrs[:-6]
                    # Log transform with offset to handle zeros
                    eps = 0.001
                    tp_cum_6hrs = np.log(eps + tp_cum_6hrs) - np.log(eps)
                    np_vars[var] = tp_cum_6hrs[-HOURS_PER_YEAR:]
                else:
                    np_vars[var] = ds[code].to_numpy()[-HOURS_PER_YEAR:]

                if partition == "train":
                    # Compute normalization statistics
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                # Compute climatological mean
                clim_yearly = np_vars[var].mean(axis=0)
                if var not in climatology:
                    climatology[var] = [clim_yearly]
                else:
                    climatology[var].append(clim_yearly)

            else:  # Pressure-level variables
                assert len(ds[code].shape) == 4
                # Process each pressure level separately
                all_levels = ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[
                        -HOURS_PER_YEAR:
                    ]

                    if partition == "train":
                        # Compute normalization statistics for each pressure level
                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        if f"{var}_{level}" not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                            normalize_std[f"{var}_{level}"] = [var_std_yearly]
                        else:
                            normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                            normalize_std[f"{var}_{level}"].append(var_std_yearly)

                    # Compute climatological mean for each pressure level
                    clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                    if f"{var}_{level}" not in climatology:
                        climatology[f"{var}_{level}"] = [clim_yearly]
                    else:
                        climatology[f"{var}_{level}"].append(clim_yearly)

        # Save data in shards
        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        # Finalize normalization statistics
        for var in normalize_mean.keys():
            if not constants_are_downloaded or var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        # Compute global statistics across years
        for var in normalize_mean.keys():
            if not constants_are_downloaded or var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # Use law of total variance to combine yearly statistics
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (
                    (std**2).mean(axis=0)  # E[var(X|Y)]
                    + (mean**2).mean(axis=0)  # E[X²]
                    - mean.mean(axis=0) ** 2  # (E[X])²
                )
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                # Special case for precipitation: center at zero
                if var == "total_precipitation":
                    normalize_mean[var] = np.zeros_like(normalize_mean[var])
                normalize_std[var] = std

        # Save normalization statistics
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    # Save climatological means
    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )


def convert_nc2npz(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    """
    Convert NetCDF (.nc) climate data files to NumPy (.npz) format with preprocessing.
    Makes use of the nc2np function to process the data.
    
    Args:
        root_dir: Root directory containing NetCDF files
        save_dir: Directory to save processed .npz files
    """
    assert (
        start_val_year > start_train_year
        and start_test_year > start_val_year
        and end_year > start_test_year
    )

    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    nc2np(root_dir, variables, train_years, save_dir, "train", num_shards)
    nc2np(root_dir, variables, val_years, save_dir, "val", num_shards)
    nc2np(root_dir, variables, test_years, save_dir, "test", num_shards)

    # save lat and lon data
    ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc"))
    x = xr.open_mfdataset(ps[0], parallel=True)
    lat = np.array(x["lat"])
    lon = np.array(x["lon"])
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)
