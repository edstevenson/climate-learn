import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_to_isobaric_levels(var_native, P_native, P_levels):
    """
    Interpolates a 4D variable (time, nlev, lat, lon) from some native pressure levels (usually derived from hybrid levels) to specified isobaric levels.
    
    Parameters:
      var_native: numpy array of shape (time, nlev, lat, lon), the variable to interpolate.
      P_native  : numpy array of shape (time, nlev, lat, lon), the computed pressure at each hybrid level.
      P_levels  : 1D numpy array of target pressure levels (in the same units as P_native).
    
    Returns:
      var_interp   : numpy array of shape (time, len(P_levels), lat, lon) with var interpolated
                  onto the target pressure levels.
    """
    time, nlev_native, nlat, nlon = var_native.shape
    nlev_target = len(P_levels)
    var_interp = np.empty((time, nlev_target, nlat, nlon))
    
    # Loop over each time step and horizontal grid point
    for t in range(time):
        for j in range(nlon):
            for i in range(nlat):
                # Extract the vertical profiles at a given time and grid point.
                P_prof = P_native[t, :, i, j]
                var_prof = var_native[t, :, i, j]
                
                # Ensure the pressure profile is monotonically increasing for interpolation.
                if P_prof[0] > P_prof[-1]:
                    P_prof = P_prof[::-1]
                    var_prof = var_prof[::-1]
                
                # Perform the 1D interpolation using numpy.interp.
                # np.interp(x, xp, fp) returns the interpolated values at x given the points xp and values fp.
                var_interp[t, :, i, j] = np.interp(P_levels, P_prof, var_prof)
    
    return var_interp

def interpolate_to_isobaric_grid(var_native, P_native, P_levels,
                                      lat_native, lon_native, 
                                      lat_target, lon_target):
    """
    Interpolates a 4D variable (time, nlev, lat, lon) from some native pressure grid (usually derived from hybrid levels) to a new target isobaric pressure grid.
    
    Parameters:
      var_native: numpy array of shape (time, nlev_native, nlat, nlon), the variable to interpolate.
      P_native  : numpy array of shape (time, nlev_native, nlat, nlon), pressure computed at each native level.
      P_levels  : 1D numpy array of target pressure levels (in same units as P_native).
      lat_native: 1D numpy array of native latitudes.
      lon_native: 1D numpy array of native longitudes.
      lat_target: 1D numpy array of target latitudes.
      lon_target: 1D numpy array of target longitudes.
    
    Returns:
      var_final : numpy array of shape (time, len(P_levels), len(lat_target), len(lon_target))
                  with the variable interpolated onto the target pressure and horizontal grid.
    """
    time, nlev_native, nlat, nlon = var_native.shape
    nlev_target = len(P_levels)
    
    # First step: Vertical interpolation.
    # Create an intermediate array with shape (time, nlev_target, nlat, nlon)
    var_vert = np.empty((time, nlev_target, nlat, nlon))
    
    for t in range(time):
        for j in range(nlat):
            for i in range(nlon):
                # Extract the vertical profile at (t, j, i)
                P_prof = P_native[t, :, j, i]
                var_prof = var_native[t, :, j, i]
                
                # Ensure the pressure profile is monotonic increasing.
                if P_prof[0] > P_prof[-1]:
                    P_prof = P_prof[::-1]
                    var_prof = var_prof[::-1]
                
                # Interpolate the vertical profile onto the target pressure levels.
                # np.interp expects the x-array to be increasing.
                var_vert[t, :, j, i] = np.interp(P_levels, P_prof, var_prof)
    
    # Second step: Horizontal interpolation.
    # We'll create a final array with shape (time, nlev_target, nlat_target, nlon_target)
    nlat_target = len(lat_target)
    nlon_target = len(lon_target)
    var_final = np.empty((time, nlev_target, nlat_target, nlon_target))
    
    # Build a meshgrid of target coordinates.
    # Note: RegularGridInterpolator expects the grid points in increasing order.
    target_pts = np.array(np.meshgrid(lat_target, lon_target, indexing='ij'))
    # Stack them so each point is (lat, lon)
    target_coords = np.column_stack((target_pts[0].ravel(), target_pts[1].ravel()))
    
    # For each time and each target vertical level, interpolate horizontally.
    for t in range(time):
        for k in range(nlev_target):
            # Build the interpolator on the native lat-lon grid.
            # The native grid is defined by lat_native and lon_native.
            # The data are from var_vert[t, k, :, :] with shape (nlat, nlon).
            interp_func = RegularGridInterpolator((lat_native, lon_native),
                                                    var_vert[t, k, :, :],
                                                    bounds_error=False, fill_value=None)
            # Evaluate the interpolator on the target coordinates.
            var_horiz = interp_func(target_coords)
            # Reshape back into (nlat_target, nlon_target)
            var_final[t, k, :, :] = var_horiz.reshape(nlat_target, nlon_target)
    
    return var_final

def interpolate_to_horizontal_grid(var_native, lat_native, lon_native, lat_target, lon_target):
    """
    Interpolates a 4D single-level variable (time, 1, lat, lon) onto a target horizontal grid.
    
    Parameters:
      var_native: numpy array of shape (time, 1, nlat, nlon), the variable to interpolate.
      lat_native: 1D numpy array of native latitudes.
      lon_native: 1D numpy array of native longitudes.
      lat_target: 1D numpy array of target latitudes.
      lon_target: 1D numpy array of target longitudes.
    
    Returns:
      var_interp: numpy array of shape (time, 1, len(lat_target), len(lon_target)) with the variable
                  interpolated onto the target horizontal grid.
    """
    assert var_native.ndim == 4 and var_native.shape[1] == 1
    
    time, _, nlat, nlon = var_native.shape
    nlat_target = len(lat_target)
    nlon_target = len(lon_target)
    
    var_interp = np.empty((time, 1, nlat_target, nlon_target))
    
    # Create a meshgrid of target coordinates.
    # RegularGridInterpolator expects grid points in increasing order.
    target_lat_grid, target_lon_grid = np.meshgrid(lat_target, lon_target, indexing='ij')
    # Stack into a (nlat_target*nlon_target, 2) array of (lat, lon) pairs.
    target_coords = np.column_stack((target_lat_grid.ravel(), target_lon_grid.ravel()))
    
    # Loop over each time step and interpolate the 2D field.
    for t in range(time):
        # Extract the 2D field (from the single level).
        field = var_native[t, 0, :, :]
        
        # Build the interpolator on the native grid.
        interp_func = RegularGridInterpolator((lat_native, lon_native),
                                                field,
                                                bounds_error=False, fill_value=None)
        # Evaluate the interpolator on the target coordinates.
        field_interp = interp_func(target_coords).reshape(nlat_target, nlon_target)
        
        # Store the result with the singleton level dimension.
        var_interp[t, 0, :, :] = field_interp
        
    return var_interp
