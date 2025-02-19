from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from climate_learn.data.processing.interpolators import interpolate_to_isobaric_grid, interpolate_to_horizontal_grid
from climate_learn.data.processing.gcm_constants import *
import os

class GCMDataset(ABC):
    def __init__(self, nc_files: list[str] | str, VAR_CODE: dict, SINGLE_LEVEL_VARS: list[str], VAR_UNIT: dict):
        """
        Base class for processing climate datasets.
        Accepts either a single filename or a list of filenames.
        If a list is provided, it uses xarray.open_mfdataset to merge the files.
        """
        self.nc_files = nc_files
        if isinstance(nc_files, (list, tuple)):
            self.ds = xr.open_mfdataset(nc_files, combine='by_coords')
        else:
            self.ds = xr.open_dataset(nc_files)
        self.VAR_CODE = VAR_CODE
        self.SINGLE_LEVEL_VARS = SINGLE_LEVEL_VARS
        self.VAR_UNIT = VAR_UNIT
        # self.CODE_VAR = {v: k for k, v in self.VAR_CODE.items()}
    
    @abstractmethod
    def compute_pressure(self):
        """
        Compute the 4D pressure field from the dataset.
        Subclasses should implement their GCM-specific pressure calculation.
        """
        pass

    def load_variable(self, var: str):
        """
        Loads a variable from the dataset using the passed constants, and unsqueezes the variables to 4D (time, level, lat, lon).
        
        Args:
            var (str): standardized variable name.
        
        Returns:
            np.ndarray: The variable's values.
        """
        data = self.ds[self.VAR_CODE[var]].values 
        data = self.VAR_UNIT[var](data)

        if var in self.SINGLE_LEVEL_VARS:
            data = data[:, None, ...]

            
        return data


    def regrid(self, var_native, P_native, P_levels, lat_native, lon_native, lat_target, lon_target):
        return interpolate_to_isobaric_grid(var_native, P_native, P_levels,
                                            lat_native, lon_native, lat_target, lon_target)
    
    def single_level_regrid(self, var_native, lat_native, lon_native, lat_target, lon_target):
        return interpolate_to_horizontal_grid(var_native, lat_native, lon_native, lat_target, lon_target)

    def close(self):
        self.ds.close()

    def to_npz(self, save_dir: str, variables: list[str],
               lat_target: np.ndarray, lon_target: np.ndarray,
               P_levels: np.ndarray, num_shards: int = 1):
        """
        Converts the dataset to NumPy files with standardized variables.

        This method:
          1. Extracts native coordinates.
          2. Loads each variable, standardizes and interpolates it to a common isobaric grid.
          3. Computes normalization statistics (mean and std) and climatology.
          4. Saves normalization stats, climatology, processed training data, and metadata.
        """
        # Create the train directory if it does not exist.
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    
        # Extract native horizontal coordinates.
        lat_native = self.load_variable("latitude")
        lon_native = self.load_variable("longitude")
    
        # Compute native pressure field once (for vertical regridding).
        P_native = self.compute_pressure()
    
        out_vars = {}
    
        # Process variables: use single-level if present, otherwise load and regrid.
        for var in variables:
            native_data = self.load_variable(var)
            
            if var in self.SINGLE_LEVEL_VARS:
                assert native_data.ndim == 4 and native_data.shape[1] == 1
                out_vars[var] = interpolate_to_horizontal_grid(native_data, lat_native, lon_native, lat_target, lon_target)

            else:
                assert native_data.ndim == 4 and native_data.shape[1] == P_native.shape[1]
                out_vars[var] = self.regrid(native_data, P_native, P_levels,
                                             lat_native, lon_native, lat_target, lon_target)
    
        # Compute normalization statistics and climatology.
        normalize_mean = {}
        normalize_std = {}
        climatology = {}
    
        for var, data in out_vars.items():
            normalize_mean[var] = data.mean(axis=(0, 2, 3))
            normalize_std[var] = data.std(axis=(0, 2, 3))
            climatology[var] = data.mean(axis=0)
    
        # Save normalization statistics.
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)
    
        # Save climatological means under the "train" subdirectory.
        np.savez(os.path.join(save_dir, "train", "climatology.npz"), **climatology)
    
        # Save the processed training data.
        total_time = next(iter(out_vars.values())).shape[0]
        if num_shards == 1:
            np.savez(os.path.join(save_dir, "train", "data.npz"), **out_vars)
        else:
            remainder = total_time % num_shards
            shard_size = total_time // num_shards
            for shard in range(num_shards):
                start_idx = shard * shard_size + min(shard, remainder)
                end_idx = start_idx + shard_size + (1 if shard < remainder else 0)
                sharded_data = {var: data[start_idx:end_idx] for var, data in out_vars.items()}
                np.savez(os.path.join(save_dir, "train", f"data_shard_{shard}.npz"), **sharded_data)
    
        # Save metadata: target latitude and longitude.
        np.save(os.path.join(save_dir, "lat.npy"), lat_target)
        np.save(os.path.join(save_dir, "lon.npy"), lon_target)
    
        self.close()


class ExoCAMDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str):
        super().__init__(nc_files, VAR_CODE_EXOCAM, SINGLE_LEVEL_VARS)

    def compute_pressure(self):
        """
        Computes the 4D pressure field for ExoCAM data using:
            P = hyam * P0 + hybm * PS

        The resulting pressure profile is returned with shape (time, lev, lat, lon)
        
        Assumes:
            hyam: 1D array of hybrid A coefficients (nlev,)
            hybm: 1D array of hybrid B coefficients (nlev,)
            P0: scalar reference pressure
            PS: 3D surface pressure array (time, lat, lon)
        
        """
        A = self.ds["hyam"].values  # Hybrid A coefficients, shape: (nlev,)
        B = self.ds["hybm"].values  # Hybrid B coefficients, shape: (nlev,)
        P0 = self.ds["P0"].values    # Reference pressure (scalar)
        PS = self.ds["PS"].values    # Surface pressure, expected shape: (time, lat, lon)
        
        P = A[None,:,None,None] * P0 + B[None,:,None,None] * PS[:,None,:,:]
        return P
    
    def compute_w(self, g, R_d, R_v):
        """
        Calculates the vertical velocity w (m/s) from the OMEGA variable.
        Calculates density as an intermediate step.

        rho = P / (R * T), R = (1-q_v) * R_d + q_v * R_v
        where q_v is the specific humidity and R_d and R_v are the gas constants for dry air and vapor, respectively.
        w = OMEGA / (rho * g) where g is surface gravity.

        Returns:
            np.ndarray: Vertical velocity w in m/s
        """
        OMEGA = self.load_variable("OMEGA")
        T = self.load_variable("temperature") 
        q_v = self.load_variable("specific_humidity")
        P = self.compute_pressure()

        R = (1-q_v) * R_d + q_v * R_v
        rho = P / (R * T)
        w = OMEGA / (rho * g)
        return w
    
    def load_variable(self, var: str, g=None, R_d=None, R_v=None):
        """
        Loads a variable from the dataset using the passed constants, and unsqueezes the variables to 4D (time, level, lat, lon).
        """
        if var == "w":
            if any(x is None for x in [g, R_d, R_v]):
                raise ValueError("Must provide g, R_d and R_v constants to compute vertical velocity w from OMEGAfor ExoCAM")
            return self.compute_w(g, R_d, R_v)
        data = super().load_variable(var)
        return data


        

class LMDGDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str):
        super().__init__(nc_files, VAR_CODE_LMDG, SINGLE_LEVEL_VARS)

    def compute_pressure(self):
        """
        Computes the 4D pressure field for LMDG data using:
            P = aps + bps * ps
        """
        aps = self.ds["aps"].values
        bps = self.ds["bps"].values
        ps = self.ds["ps"].values

        P = aps[None,:,None,None] + bps[None,:,None,None] * ps[:,None,:,:]
        return P


class ROCKE3DDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str):
        super().__init__(nc_files, VAR_CODE_ROCKE3D, SINGLE_LEVEL_VARS)

    def compute_pressure(self):
        """
        pressure is a diagnostic variable in ROCKE3D:
        """
        P = self.ds["p_3d"].values * 1e2 # convert to Pa (from hPa = mbar)
        return P


class UMDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str):
        super().__init__(nc_files, VAR_CODE_UM, SINGLE_LEVEL_VARS)

    def compute_pressure(self):
        """
        Pressure is a diagnostic variable in UM. But need to distinguish between rho and theta levels.
        Returns a tuple: (P_rho, P_theta)
        """
        P_rho = self.ds["pressure_at_rho_levels"].values
        P_theta = self.ds["pressure_at_theta_levels"].values
        return P_rho, P_theta

    def to_npz(self, save_dir: str, variables: list[str],
               lat_target: np.ndarray, lon_target: np.ndarray,
               P_levels: np.ndarray, num_shards: int = 1):
        """
        Converts the UM dataset to NumPy files with standardized variables.

        For UM:
          - 'u' and 'v' are regridded using the theta pressure field and their specific horizontal grids:
             * 'u' uses variables "latitude_u" and "longitude_u".
             * 'v' uses variables "latitude_v" and "longitude_v".
          - 'temperature' is regridded using the theta pressure field and the temperature grid
             ("latitude_t" and "longitude_t").
          - Other variables use the default horizontal grid (from self.load_variable("latitude"))
            and the rho pressure field.
        """
        # Create the train directory if it does not exist.
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)

        # Compute native pressure fields.
        P_rho, P_theta = self.compute_pressure()

        out_vars = {}
        for var in variables:
            native_data = self.load_variable(var)

            if var in self.SINGLE_LEVEL_VARS:
                assert native_data.ndim == 4 and native_data.shape[1] == 1
                lat_native = self.load_variable("latitude_t")
                lon_native = self.load_variable("longitude_t")
                out_vars[var] = self.single_level_regrid(native_data, lat_native, lon_native, lat_target, lon_target)

            else:
                assert native_data.ndim == 4 and native_data.shape[1] == P_rho.shape[1]
                
                if var in ["u", "v"]:
                    lat_native = self.load_variable(f"latitude_{var}")
                    lon_native = self.load_variable(f"longitude_{var}")
                    P_native = P_rho

                elif var in ["w", "temperature", "heating_sw", "heating_lw", "specific_humidity", "relative_humidity", "cloud_ice_mmr", "cloud_liquid_mmr", "cloud_fraction"]:
                    lat_native = self.load_variable("latitude_t")
                    lon_native = self.load_variable("longitude_t")
                    P_native = P_theta

                else:
                    raise NotImplementedError(f"Is {var} defined on the rho or theta levels? which horizontal grid?")

                out_vars[var] = self.regrid(native_data, P_native, P_levels,
                                            lat_native, lon_native, lat_target, lon_target)

        # Compute normalization statistics and climatology.
        normalize_mean = {}
        normalize_std = {}
        climatology = {}
        for var, data in out_vars.items():
            normalize_mean[var] = data.mean(axis=(0, 2, 3))
            normalize_std[var] = data.std(axis=(0, 2, 3))
            climatology[var] = data.mean(axis=0)

        # Save normalization statistics.
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

        # Save climatological means under the "train" subdirectory.
        np.savez(os.path.join(save_dir, "train", "climatology.npz"), **climatology)

        # Save the processed training data.
        total_time = next(iter(out_vars.values())).shape[0]
        if num_shards == 1:
            np.savez(os.path.join(save_dir, "train", "data.npz"), **out_vars)
        else:
            remainder = total_time % num_shards
            shard_size = total_time // num_shards
            for shard in range(num_shards):
                start_idx = shard * shard_size + min(shard, remainder)
                end_idx = start_idx + shard_size + (1 if shard < remainder else 0)
                sharded_data = {var: data[start_idx:end_idx] for var, data in out_vars.items()}
                np.savez(os.path.join(save_dir, "train", f"data_shard_{shard}.npz"), **sharded_data)

        # Save metadata: target latitude and longitude arrays.
        np.save(os.path.join(save_dir, "lat.npy"), lat_target)
        np.save(os.path.join(save_dir, "lon.npy"), lon_target)

        self.close()

    