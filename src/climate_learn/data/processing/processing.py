from abc import ABC
import numpy as np
if not hasattr(np, 'round_'): # Monkey patch 
    np.round_ = np.round
import xgcm as xg
import xarray as xr
import xesmf as xe
from climate_learn.data.processing.gcm_constants import *
import os
from rich.traceback import install
install()



class GCMDataset(ABC):
    def __init__(self, nc_files: list[str] | str, VAR_CODE: dict, VAR_UNIT: dict, SINGLE_LEVEL_VARS: list[str], lon_substellar: float = 0.0):
        """
        Base class for processing climate datasets.
        Accepts either a single filename or a list of filenames which are concatenated along the time dimension.
        If a list is provided, it uses xarray.open_mfdataset to merge the files.
        lon_substellar: float = 0.0, # lon of substellar point in dataset (degrees)
        """
        self.nc_files = nc_files
        self.VAR_CODE = VAR_CODE
        self.SINGLE_LEVEL_VARS = SINGLE_LEVEL_VARS
        self.VAR_UNIT = VAR_UNIT
        self.lon_substellar = lon_substellar
        
        if isinstance(nc_files, (list, tuple)):
            self.ds = xr.open_mfdataset(nc_files, combine='nested', concat_dim=VAR_CODE["time"], decode_times=False)
        else:
            self.ds = xr.open_dataset(nc_files, decode_times=False)

    def load_variable(self, var: str):
        """
        Loads a variable from the dataset using the passed constants, and unsqueezes the variables to 4D (time, level, lat, lon).
        """
        # Get the variable code from the passed constants, or use the variable name if not found.
        code = self.VAR_CODE.get(var, var) 
        data = self.ds[code].values
        
        # Apply unit conversion if available; if not, return the raw data.
        conversion_fn = self.VAR_UNIT.get(var, lambda x: x)
        data = conversion_fn(data)

        if var in self.SINGLE_LEVEL_VARS:
            assert data.ndim == 3, f"{var} has shape {data.shape}, are you sure it's a single-level variable?"
            data = data[:, None, ...]
            
        return data

    def regrid_lat_lon(self, 
                       lat_target: np.ndarray, 
                       lon_target: np.ndarray):
        """
        Regrid dataset to target latitude and longitude.
        Note: assumes the horizontal dimensions are the last/rightmost dimensions.
        """
        target_grid = xr.Dataset(
            {"lat": (["lat"], lat_target),
            "lon": (["lon"], lon_target)},
        )
        # Rename coordinate names to "lat" and "lon" (xESMF requires these names).
        ds = self.ds.rename({self.VAR_CODE["latitude"]: "lat", self.VAR_CODE["longitude"]: "lon"})

        # Check if the longitude coordinate is duplicated. If so, drop the last value.
        if np.allclose(ds["lon"].values[0] % 360, ds["lon"].values[-1] % 360):
            ds = ds.isel(lon=slice(0, -1))

        # If a longitude shift is specified (nonzero), adjust the lon coordinate.
        if self.lon_substellar:
            ds = ds.assign_coords(lon=((ds["lon"] - self.lon_substellar)))

        regridder = xe.Regridder(ds, target_grid, method="bilinear", periodic=True)
        
        ds_out = regridder(ds, keep_attrs=False)

        self.ds = ds_out
        return self.ds
    
    def regrid_levels(self, P_levels: np.ndarray):
        """
        Regrid dataset to the provided isobaric pressure levels P_levels.
        Uses log-linear interpolation via xgcm.
        not-4D variables are left unchanged.

        Returns:
            xr.Dataset: The dataset with variables regridded along the vertical axis.
        """
        # Create the xgcm grid using the vertical coordinate 
        grid = xg.Grid(self.ds, coords={'Z': {'center': self.VAR_CODE["level"]}}, periodic=False)

        ds_out = xr.Dataset(coords=self.ds.coords)
        pressure_str = self.VAR_CODE.get('pressure', 'pressure')
        # Iterate over each data variable in the dataset
        for code, da in self.ds.data_vars.items():
            if self.VAR_CODE["level"] in da.dims and da.ndim == 4 and code != pressure_str:
                # Apply vertical regridding using log-linear interpolation.
                transformed = grid.transform(
                    da,
                    'Z',
                    P_levels,
                    target_data=self.ds[pressure_str],
                    method="log"
                )
                # xgcm by default puts the new vertical axis at the end, e.g.
                # (time, lat, lon, vertical). We want (time, vertical, lat, lon).
                transformed = transformed.transpose(transformed.dims[0],
                                                    transformed.dims[-1],
                                                    *transformed.dims[1:-1])
                ds_out[code] = transformed
            elif code == pressure_str:
                # Remove the pressure variable to avoid confusion.
                continue
            else:
                # Keep unchanged variables that do not have the vertical level dimension.
                ds_out[code] = da
        
        # Rename the pressure coordinate  to 'isobar' (sometimes used for derived variables)
        ds_out = ds_out.rename_dims({pressure_str: "isobar"}).rename_vars({pressure_str: "isobar"})
        self.ds = ds_out
        return self.ds
    
    def regrid(self, P_levels: np.ndarray, lat_target: np.ndarray, lon_target: np.ndarray):
        """
        Regrid dataset to the provided isobaric pressure levels P_levels, and target latitude and longitude.
        """
        self.regrid_levels(P_levels)
        self.regrid_lat_lon(lat_target, lon_target)
        return self.ds
        
    def close(self):
        self.ds.close()

    def to_npz(self, save_dir: str, variables: list[str],
               lat_target: np.ndarray, lon_target: np.ndarray,
               P_levels: np.ndarray, shard_size: int | None = None, progress_bar: bool = False):
        """
        Converts the dataset to NumPy files with standardized variables.

        This method:
          1. Extracts native coordinates.
          2. Loads each variable, standardizes and interpolates it to a common isobaric grid.
          3. Computes normalization statistics (mean and std) and climatology.
          4. Saves normalization stats, climatology, processed training data, and metadata.
          
        Sharding:
          If `shard_size` is provided and less than the total time dimension,
          the data is split into shards of fixed size (the last shard contains the remainder).
          If `shard_size` is None or larger than the total time dimension, the entire data is saved in one file.
        """
        # Regrid dataset to isobaric pressure grid.
        self.regrid(P_levels, lat_target, lon_target)

        # load variables as numpy arrays
        if progress_bar:
            print("Loading variables...")
            from dask.diagnostics import ProgressBar
            # will only show progress bar if using dask (i.e., if dataset is chunked)
            with ProgressBar():
                out_vars = {var: self.load_variable(var) for var in variables}
        else:
            out_vars = {var: self.load_variable(var) for var in variables}

        # Compute normalization statistics and climatology.
        normalize_mean = {}
        normalize_std = {}
        climatology = {}
        for var, data in out_vars.items():
            normalize_mean[var] = data.mean(axis=(0, 2, 3))
            normalize_std[var] = data.std(axis=(0, 2, 3))
            climatology[var] = data.mean(axis=0)

        # Save the processed training data.
        # Create the train directory if it does not exist.
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
    
        # Save normalization statistics.
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)
    
        # Save climatological means under the "train" subdirectory.
        np.savez(os.path.join(save_dir, "train", "climatology.npz"), **climatology)
    
        total_time = next(iter(out_vars.values())).shape[0]
        if shard_size is None or total_time <= shard_size:
            # Save entire data in one file if no shard_size provided or data fits in one shard.
            np.savez(os.path.join(save_dir, "train", "data.npz"), **out_vars)
        else:
            num_shards = (total_time + shard_size - 1) // shard_size  # ceiling division
            for shard in range(num_shards):
                start_idx = shard * shard_size
                end_idx = min((shard + 1) * shard_size, total_time)
                sharded_data = {var: data[start_idx:end_idx] for var, data in out_vars.items()}
                np.savez(os.path.join(save_dir, "train", f"data_shard_{shard}.npz"), **sharded_data)
    
        # Save metadata: target latitude and longitude.
        np.save(os.path.join(save_dir, "lat.npy"), lat_target)
        np.save(os.path.join(save_dir, "lon.npy"), lon_target)

        print(f'\033[1mnp data saved to {save_dir}\033[0m')
    
        self.close()


class ExoCAMDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str, 
                 g: float | None = None, 
                 R_d: float | None = None, 
                 R_v: float | None = None, 
                 lon_substellar: float = 0.0):
        # ExoCAM THAI data seems to use 180 as the substellar point
        super().__init__(nc_files, VAR_CODE_EXOCAM, VAR_UNIT_EXOCAM, SINGLE_LEVEL_VARS, lon_substellar)
        self.g = g       
        self.R_d = R_d  
        self.R_v = R_v
        self.add_pressure()   # Add pressure to the dataset for use in regridding

    def add_pressure(self):
        """
        Computes the 4D pressure field for ExoCAM data using:
            P = hyam * P0 + hybm * PS
        and adds it to the dataset as a new DataArray labelled "pressure".
        
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

        # Compute pressure with shape (time, nlev, lat, lon)
        P = A[None, :, None, None] * P0 + B[None, :, None, None] * PS[:, None, :, :]

        # Add the computed pressure as a new DataArray to the dataset.
        # Use dimensions from VAR_CODE if available, otherwise default to ("time", "lev", "lat", "lon")
        self.ds["pressure"] = xr.DataArray(
            P,
            dims=(
                self.VAR_CODE["time"],
                self.VAR_CODE["level"],
                self.VAR_CODE["latitude"],
                self.VAR_CODE["longitude"]
            )
        )
    
    def compute_w(self):
        """
        Calculates the vertical velocity w (m/s) from the OMEGA variable.
        Calculates density as an intermediate step.

        rho = P / (R * T), R = (1-q_v) * R_d + q_v * R_v
        where q_v is the specific humidity and R_d and R_v are the gas constants for dry air and vapor, respectively.
        w = OMEGA / (rho * g) where g is surface gravity.

        Returns:
            np.ndarray: Vertical velocity w in m/s
        """
        if any(x is None for x in [self.g, self.R_d, self.R_v]):
                raise ValueError("Must provide g, R_d and R_v constants to ExoCAM constructor to compute vertical velocity w from OMEGA")
        
        OMEGA = super().load_variable("OMEGA")
        T = super().load_variable("temperature") 
        q_v = super().load_variable("specific_humidity")
        P = self.load_variable("isobar") # assumes regrid_levels has already been called
        R = (1-q_v) * self.R_d + q_v * self.R_v
        rho = P[None, :, None, None] / (R * T)
        w = OMEGA / (rho * self.g)
        return w
    
    def load_variable(self, var: str):
        """
        Loads a variable from the dataset using the passed constants, and unsqueezes the variables to 4D (time, level, lat, lon).
        """
        if var == "w":
            return self.compute_w()
        return super().load_variable(var)
      

class LMDGDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str, lon_substellar: float = 0.0):
        super().__init__(nc_files, VAR_CODE_LMDG, VAR_UNIT_LMDG, SINGLE_LEVEL_VARS, lon_substellar)


class ROCKE3DDataset(GCMDataset):
    def __init__(self, nc_files: list[str], lon_substellar: float = 0.0):
        # merge the various radiation and other-variable files  
        self.ds = xr.merge([xr.open_dataset(f, decode_times=False) for f in nc_files])
        
        self.VAR_CODE = VAR_CODE_ROCKE3D
        self.VAR_UNIT = VAR_UNIT_ROCKE3D
        self.SINGLE_LEVEL_VARS = SINGLE_LEVEL_VARS
        self.lon_substellar = lon_substellar

    def load_variable(self, var: str):
        if var == "asr_clear":
            incident_sw_top_of_atmo = self.load_variable("incident_sw_top_of_atmo")
            reflected_sw_top_of_atmo_clear = self.load_variable("reflected_sw_top_of_atmo_clear")
            asr_clear = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_clear
            return asr_clear
        
        if var == "olr_cloudy":
            olr_clear = self.load_variable("olr_clear")
            lw_cloud_rad_forcing = self.load_variable("lw_cloud_rad_forcing")
            olr_cloudy = olr_clear - lw_cloud_rad_forcing
            return olr_cloudy
        
        if var == "surface_lw_net":
            surface_lw_down = self.load_variable("surface_lw_down")
            surface_lw_up = self.load_variable("surface_lw_up")
            surface_lw_net = surface_lw_down - surface_lw_up
            return surface_lw_net

        return super().load_variable(var)

class UMDataset(GCMDataset):
    def __init__(self, nc_files: list[str] | str, lon_substellar: float = 0.0):
        
        self.nc_files = nc_files
        self.VAR_CODE = VAR_CODE_UM
        self.SINGLE_LEVEL_VARS = SINGLE_LEVEL_VARS
        self.VAR_UNIT = VAR_UNIT_UM
        self.lon_substellar = lon_substellar
        
        time_rad = VAR_CODE_UM["time_rad"]    
        time = VAR_CODE_UM["time"]

        def merge_time_coords(ds):
                # Identify all data variables that have time_rad in their dimensions.
                rad_vars = [var for var in ds.data_vars if time_rad in ds[var].dims]
                
                # Also include the time_rad coordinate 
                if time_rad in ds.coords: rad_vars.append(time_rad)

                ds_rad = ds[rad_vars]
                ds_other = ds.drop_vars(rad_vars, errors="ignore").drop_dims(time_rad, errors="ignore")

                # Rename the 'time_rad' dimension and coordinate to 'time' in the rad-dependent dataset.
                ds_rad = ds_rad.rename_dims({time_rad: time})
                if time_rad in ds_rad.coords:
                    ds_rad = ds_rad.rename({time_rad: time})


                # Instead of using xr.merge (which may try to align or concatenate along shared dimensions),
                # update ds_other by simply adding the variables from ds_rad.
                ds_combined = ds_other.copy()
                for var in ds_rad.data_vars:
                    ds_combined[var] = ds_rad[var]
                
                return ds_combined
        
        if isinstance(nc_files, (list, tuple)):           
            ds = xr.open_mfdataset(nc_files,
                                    combine="nested",
                                    decode_times=False,
                                    preprocess=merge_time_coords, 
                                    concat_dim=time)
        else:
            ds = xr.open_dataset(nc_files, decode_times=False)
            ds = merge_time_coords(ds)

        # Rechunk the entire dataset along the time dimension, with the vertical dimensions unchunked.
        self.ds = ds.chunk({time: 256, self.VAR_CODE["theta_level"]: -1, self.VAR_CODE["rho_level"]: -1})
        print(self.ds.chunks)


    def load_variable(self, var: str):
        if var == "asr_clear":
            incident_sw_top_of_atmo = self.load_variable("incident_sw_top_of_atmo")
            reflected_sw_top_of_atmo_clear = self.load_variable("reflected_sw_top_of_atmo_clear")
            asr_clear = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_clear
            return asr_clear
        
        if var == "asr_cloudy":
            incident_sw_top_of_atmo = self.load_variable("incident_sw_top_of_atmo")
            reflected_sw_top_of_atmo_cloudy = self.load_variable("reflected_sw_top_of_atmo_cloudy")
            asr_cloudy = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_cloudy
            return asr_cloudy

        return super().load_variable(var)

    def regrid_uv(self):
        """
        Regrid horizontal velocities u, v, defined on their own separate horizontal grids to the target ('t') grid.
        
        Returns:
            xr.Dataset: The dataset with regridded u and v variables on the t grid.
        """
        # get codes
        u_code = self.VAR_CODE.get("u", "u")
        v_code = self.VAR_CODE.get("v", "v")
        u_lat_code = self.VAR_CODE.get("latitude_u", "latitude_u")
        u_lon_code = self.VAR_CODE.get("longitude_u", "longitude_u")
        v_lat_code = self.VAR_CODE.get("latitude_v", "latitude_v")
        v_lon_code = self.VAR_CODE.get("longitude_v", "longitude_v")

        # Rename the source horizontal coordinate names to "lat" and "lon" for xESMF.
        u_da = self.ds[u_code].rename({u_lat_code: "lat", u_lon_code: "lon"})  
        v_da = self.ds[v_code].rename({v_lat_code: "lat", v_lon_code: "lon"})  

        # Create the target grid using the t grid coordinates.
        target_grid = xr.Dataset({
            "lat": (["lat"], self.ds[self.VAR_CODE["latitude"]].values),
            "lon": (["lon"], self.ds[self.VAR_CODE["longitude"]].values)
        })
        
        # Check if the source "lon" coordinate has duplicated endpoints. If so, drop the last value.
        if np.allclose(u_da["lon"].values[0] % 360, u_da["lon"].values[-1] % 360):
            u_da = u_da.isel(lon=slice(0, -1))
        if np.allclose(v_da["lon"].values[0] % 360, v_da["lon"].values[-1] % 360):
            v_da = v_da.isel(lon=slice(0, -1))
        
        # Create the regridder with xESMF and apply it.
        u_regridder = xe.Regridder(u_da, target_grid, method="bilinear", periodic=True)
        u_regridded_da = u_regridder(u_da, keep_attrs=True)

        v_regridder = xe.Regridder(v_da, target_grid, method="bilinear", periodic=True)
        v_regridded_da = v_regridder(v_da, keep_attrs=True)

        # Rename the regridded coordinates back to that of the t grid to work with regrid_levels
        u_regridded_da = u_regridded_da.rename({"lat": self.VAR_CODE["latitude"], "lon": self.VAR_CODE["longitude"]})
        v_regridded_da = v_regridded_da.rename({"lat": self.VAR_CODE["latitude"], "lon": self.VAR_CODE["longitude"]})
        
        self.ds = self.ds.drop_vars([u_code, v_code])
        self.ds[u_code] = u_regridded_da
        self.ds[v_code] = v_regridded_da
        return self.ds
    
    def regrid_levels(self, P_levels: np.ndarray):
        """
        Regrid the UM dataset to the provided isobaric pressure levels P_levels.
        Need to account for staggered vertical grid here (rho and theta levels)
        """
        # get codes
        theta_level_str = self.VAR_CODE.get("theta_level", "theta_level")
        rho_level_str = self.VAR_CODE.get("rho_level", "rho_level")

        # Create the xgcm grids using the vertical coordinate 
        theta_grid = xg.Grid(self.ds, coords={'Z': {'center': theta_level_str}}, periodic=False)
        rho_grid = xg.Grid(self.ds, coords={'Z': {'center': rho_level_str}}, periodic=False)

        ds_out = xr.Dataset(coords=self.ds.coords)
        theta_pressure_str = self.VAR_CODE.get('pressure_at_theta_levels', 'pressure_at_theta_levels')
        rho_pressure_str = self.VAR_CODE.get('pressure_at_rho_levels', 'pressure_at_rho_levels')

        # Iterate over each data variable in the dataset
        for code, da in self.ds.data_vars.items():
            if theta_level_str in da.dims and da.ndim == 4 and code != theta_pressure_str:
                # Apply vertical regridding using log-linear interpolation.
                transformed = theta_grid.transform(
                    da,
                    'Z',
                    P_levels,
                    target_data=self.ds[theta_pressure_str],
                    method="log"
                )
                # xgcm by default puts the new vertical axis at the end, e.g.
                # (time, lat, lon, vertical). We want (time, vertical, lat, lon).
                transformed = transformed.transpose(transformed.dims[0],
                                                    transformed.dims[-1],
                                                    *transformed.dims[1:-1])
                ds_out[code] = transformed
            elif rho_level_str in da.dims and da.ndim == 4 and code != rho_pressure_str:
                transformed = rho_grid.transform(
                    da,
                    'Z',
                    P_levels,
                    target_data=self.ds[rho_pressure_str],
                    method="log"
                )
                transformed = transformed.transpose(transformed.dims[0],
                                                    transformed.dims[-1],
                                                    *transformed.dims[1:-1])
                ds_out[code] = transformed
            # remove the pressure variables to avoid confusion
            elif code == theta_pressure_str:
                # Keep the lowest level of theta-level pressure as the surface pressure
                ds_out["surface_pressure"] = self.ds[code].isel({theta_level_str: 0})
            elif code == rho_pressure_str:
                continue
            else:
                # Keep unchanged variables that do not have the vertical level dimension.
                ds_out[code] = da

        # rename pressure coordinates (mainly useful for debugging. Note: both coordinates should have the same values [xarray doesn't allow them to be named the same however])
        ds_out = ds_out.rename_dims({rho_pressure_str: "rho_isobar"}).rename_vars({rho_pressure_str: "rho_isobar"})
        ds_out = ds_out.rename_dims({theta_pressure_str: "theta_isobar"}).rename_vars({theta_pressure_str: "theta_isobar"})
        
        self.ds = ds_out
        return self.ds

    def regrid(self, P_levels: np.ndarray, lat_target: np.ndarray, lon_target: np.ndarray):
        """
        Regrid the UM dataset to the provided isobaric pressure levels P_levels, and target latitude and longitude.

        For UM:
          - u (v) are:
            - horizontally regridded from the lat/lon u (v) grid to the t grid (simple midpoint averages here) using the regrid_uv method
            - vertically regridded to the rho pressure levels
            - horizontally regridded to the lat/lon target grid
          - all other variables are on the t grid and theta levels:
            - vertically regridded to the theta pressure levels
            - horizontally regridded to the lat/lon target grid
            - (single level variables are on the t grid so are just horizontally regridded)

        Sharding:
          The processed training data is split into chunks of fixed size (determined by `shard_size`).
          If `shard_size` is None or larger than the total time dimension, all data is saved in one file.
        """
        self.regrid_uv()
        self.regrid_levels(P_levels)
        self.regrid_lat_lon(lat_target, lon_target)
        return self.ds




    