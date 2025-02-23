# ------------------- across all GCMs ------------------- #
# TODO: add geopotential height / height (needed to express pressure variations?)
SINGLE_LEVEL_VARS = [
    "olr_clear",
    "olr_cloudy",
    "asr_clear",
    "asr_cloudy",
    "surface_temperature",
    "surface_sw_down",
    "surface_lw_net",
    "surface_lw_down",
    "surface_lw_up",
    # "surface_albedo",
    "surface_pressure",
    # "open_ocean_fraction", # TODO: these are currently not implemented in processing.py 
    # "ice_fraction",
    "reflected_sw_top_of_atmo_clear",
    "reflected_sw_top_of_atmo_cloudy",
    "incident_sw_top_of_atmo",
    "lw_cloud_rad_forcing",
]

STANDARDIZED_UNITS = { # units Im choosing as the standard 
    "olr_clear": "W/m^2", # Net longwave flux at top of model (clear-sky)
    "olr_cloudy": "W/m^2", # Net longwave flux at top of model (all-sky)
    "asr_clear": "W/m^2", # Net solar flux at top of model (clear-sky)
    "asr_cloudy": "W/m^2", # Net solar flux at top of model (all-sky)
    "surface_temperature": "K",
    "surface_sw_down": "W/m^2",
    "surface_lw_net": "W/m^2",
    "cloud_liquid_path": "g/m^2",
    "cloud_ice_path": "g/m^2",
    "vapor_column": "kg/m^2",
    "temperature": "K",
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "heating_sw": "K/s", # derived quantities, maybe one to leave out of model?
    "heating_lw": "K/s",
    "specific_humidity": "1", # dimensionless (kg/kg)
    "relative_humidity": "1",
    "cloud_fraction": "1",
    "cloud_ice_mmr": "1", # dimensionless (kg/kg)
    "cloud_liquid_mmr": "1", # dimensionless (kg/kg)
}



# ------------------- EXOCAM ------------------- #

VAR_CODE_EXOCAM = {
    "olr_clear": "FLNTC", # note: doesn't exist for dry cases like Ben1
    "olr_cloudy": "FLNT", 
    "asr_clear": "FSNTC", # note: doesn't exist for dry cases like Ben1
    "asr_cloudy": "FSNT", 
    "surface_temperature": "TS", 
    "surface_pressure": "PS", 
    "surface_sw_down": "FSDS", 
    "surface_lw_net": "FLNS",
    # the commented out variables im guessing are mostly redundant - the column-integrated variables are straightforwardly derived from the full 3D fields like specific humidity, cloud ice, cloud liquid, etc...
    # "ice_fraction": "ICEFRAC", # fraction of open ocean that is ice
    # "cloud_liquid_path": "TGCLDLWP", # total cloud liquid water in a vertical column of atmo
    # "cloud_ice_path": "TGCLDIWP", # equivalent for water ice
    # "vapor_column": "TMQ",# equivalent for water vapor
    "temperature": "T",
    "u": "U",
    "v": "V",
    "OMEGA": "OMEGA", # w = OMEGA/(dp/dz) = -OMEGA/(rho*g)
    "heating_sw": "QRS",
    "heating_lw": "QRL",
    "specific_humidity": "Q",
    "relative_humidity": "RELHUM",
    "cloud_fraction": "CLOUD",
    "cloud_ice_mmr": "CLDICE",
    "cloud_liquid_mmr": "CLDLIQ",
    "latitude": "lat",
    "longitude": "lon",
    "level": "lev",
    "time": "time",
}

# Add unit conversions (to the standard units at top of this file) functions here. In processing.py, if no conversion is present, the data is returned as is.
VAR_UNIT_EXOCAM = {
    "relative_humidity":   lambda x: x/100,     # "%" → "1"
    "cloud_fraction":      lambda x: x/100,     # "%" → "1"
}

# ------------------- LMDG ------------------- #

VAR_CODE_LMDG = {
    "olr_clear": "OLRcs",
    "olr_cloudy": "OLR",
    "asr_clear": "ASRcs",
    "asr_cloudy": "ASR",
    "surface_temperature": "tsurf",
    "surface_pressure": "ps",
    "pressure": "p",
    "surface_sw_down": "fluxsurfsw",
    "surface_lw_net": "netfluxsurflw",
    # "ice_fraction": "h2o_ice_surf", # multiplied by 1000 for some reason...
    # "cloud_liquid_path": "h2o_ice_col", # TODO: check. why duplicate?
    # "cloud_ice_path": "h2o_ice_col",
    # "vapor_column": "h2o_vap_col",
    "temperature": "temp",
    "u": "u",
    "v": "v",
    "w": "w",
    "heating_sw": "zdtsw",
    "heating_lw": "zdtlw",
    "specific_humidity": "h2o_vap",
    "cloud_fraction": "CLF",
    "cloud_ice_mmr": "h2o_ice",
    # "cloud_liquid_mmr": "?",
    # "surface_albedo": "alb_surf",
    "latitude": "latitude",
    "longitude": "longitude",
    "level": "altitude",
    "time": "Time"
}

VAR_UNIT_LMDG = {}

# ------------------- ROCKE3D ------------------- #

VAR_CODE_ROCKE3D = {
    "olr_clear": "olrcs",  # Outgoing LW radiation at TOA (clear-sky)
    "lw_cloud_rad_forcing": "lwcrf_toa", 
    # olr_cloudy = olrcs - lwcrf_toa (check sign)
    "reflected_sw_top_of_atmo_clear": "swup_toa_clrsky", 
    "incident_sw_top_of_atmo": "incsw_toa",
    # asr_clear = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_clear
    "asr_cloudy": "srnf_toa",        # NET solar radiation at TOA (all-sky)
    "specific_humidity": "q",
    "relative_humidity": "rh",
    "cloud_ice_mmr": "icmmr",
    "cloud_liquid_mmr": "wcmmr",
    "cloud_ice_fraction": "icf",
    "cloud_liquid_fraction": "wcf",
    # cloud_fraction is computed from cloud_ice_fraction and cloud_liquid_fraction with some overlap assumptions that i don't seem to have access to
    "surface_temperature": "tgrnd",  
    "surface_pressure": "p_surf",    
    "surface_sw_down": "swds",      
    "surface_lw_down": "lwds",
    "surface_lw_up": "lwus",
    # "surface_lw_net": None,          # Not directly available (typically computed as lwds - lwus)
    # "surface_albedo": None,          # No albedo variable available in ROCKE3D data

    "temperature": "t",    # 3D temperature
    "pressure": "p_3d",
    "u": "u",              # east-west wind component
    "v": "v",              # north-south wind component
    "w": "w",              # vertical velocity
    "heating_sw": "swhr",  # Shortwave radiative heating rate (units: K/day)
    "heating_lw": "lwhr",  # Longwave radiative heating rate (units: K/day)
    "latitude": "lat",
    "longitude": "lon",
    "level": "level",
    "time": "time"
}

VAR_UNIT_ROCKE3D = {
    "surface_temperature":         lambda x: x + 273.15,     # "C" → "K"
    "surface_pressure":            lambda x: x * 100,        # "mbar" → "Pa"
    "pressure":                    lambda x: x * 100,        # "mbar" → "Pa"
    "heating_sw":                  lambda x: x * 86400,      # "K/day" → "K/s"
    "heating_lw":                  lambda x: x * 86400,      # "K/day" → "K/s"
    "relative_humidity":           lambda x: x/100,          # "%" → "1"
}

# ------------------- UM ------------------- #

VAR_CODE_UM = {
    "olr_clear": "STASH_m01s02i206",  # CLEAR-SKY (II) UPWARD LW FLUX (TOA)
    "olr_cloudy": "STASH_m01s02i205",  # OUTGOING LW RAD FLUX (TOA):CORRECTED (all-sky)
    "incident_sw_top_of_atmo": "STASH_m01s01i207",   
    "reflected_sw_top_of_atmo_clear": "STASH_m01s01i209",    
    # asr_clear = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_clear
    "reflected_sw_top_of_atmo_cloudy": "STASH_m01s01i205", # ie. corrected for clouds
    # asr_cloudy = incident_sw_top_of_atmo - reflected_sw_top_of_atmo_cloudy

    "surface_temperature": "STASH_m01s00i024",  # SURFACE TEMPERATURE AFTER TIMESTEP

    "surface_sw_down": "STASH_m01s01i235",       # TOTAL DOWNWARD SURFACE SW FLUX
    "surface_lw_net": "STASH_m01s02i201",         # NET DOWNWARD SURFACE LW RAD FLUX

    "temperature": "STASH_m01s16i004",             # TEMPERATURE ON THETA LEVELS

    "u": "STASH_m01s00i002",                       # U component of wind after timestep
    "v": "STASH_m01s00i003",                       
    "w": "STASH_m01s00i150",                       
    "heating_sw": "STASH_m01s01i232",              # (4D) SW HEATING RATES: ALL TIMESTEPS (K s-1)
    "heating_lw": "STASH_m01s02i232",              # (4D) LW HEATING RATES (K s-1)

    "specific_humidity": "STASH_m01s00i010",       # SPECIFIC HUMIDITY AFTER TIMESTEP (theta grid)
    "relative_humidity": "STASH_m01s30i113",       # RELATIVE HUMIDITY AFTER TIMESTEP (theta grid)

    # moisture variables
    "cloud_ice_mmr": "STASH_m01s00i012",           # QCF AFTER TIMESTEP (mass fraction of cloud ice) (theta grid)
    "cloud_liquid_mmr": "STASH_m01s00i254",        # QCL AFTER TIMESTEP (mass fraction of cloud liquid water) (theta grid)
    "cloud_fraction": "STASH_m01s00i266",           # BULK CLOUD FRACTION IN EACH LAYER (theta grid)

    "open_ocean_fraction": "STASH_m01s03i395",     # OPEN OCEAN FRACTION (theta grid)

    # 't' horizontal grid - this is the 'standard' grid for most variables and is equivalent to the lat/lon grid of the other non-staggered GCM grids
    "latitude": "latitude_t",                      
    "longitude": "longitude_t",

    # u horizontal grid
    "latitude_u": "latitude_cu",
    "longitude_u": "longitude_cu",

    # v horizontal grid
    "latitude_v": "latitude_cv",
    "longitude_v": "longitude_cv",

    # pressure at rho and theta levels
    "pressure_at_rho_levels": "STASH_m01s00i407",
    "pressure_at_theta_levels": "STASH_m01s00i408",

    # levels
    "theta_level": "thlev_eta_theta",
    "rho_level": "rholev_eta_rho",

    "time": "hourly",
    "time_rad": "hourly_rad",

    # sea level
    "zsea_rho": "rholev_zsea_rho",
    "zsea_theta": "thlev_zsea_theta",
    # because of the staggered grid, surface pressure is exactly equivalent to the bottom level value of pressure_at_theta_levels (other GCMs use grid midpoints for pressure hence the separate surface pressure variable)... I think...
}

VAR_UNIT_UM = {}








