PLANET_PARAMS = {
    "trappist-1e": {
        "radius": 5798.0, # km 
        "gravity": 9.12, # m/s^2
        "orbital period": 6.1, # (earth) days
        "rotation period": 6.1, # (earth) days
        "instellation": 900.0, # W/m^2
    }
}

CASE_PARAMS = {
    "ben1": {
        "albedo": 0.3, 
        "R_d": 297.0, # J/(kg*K) TODO: these are placeholder values -- can correct from input file when found (so are the other cases below)
        # placeholder calcs:
        # R_d = R_univ / mu_d; R_univ = 8.3145 J/(K*mol), mu_d = 28.0 g/mol
        # R_v = R_univ / mu_v; mu_v = 18.0 g/mol    
        "R_v": 461.9, # J/(kg*K) (water)
    },
    "ben2": {
        "albedo": 0.3,
        "R_d": 189.0, # J/(kg*K) mu_d = 44.0 g/mol (CO2)
        "R_v": 461.9, # J/(kg*K)
    },
    "hab1": {
        # "albedo": 0.3, ?
        "R_d": 297.0, # J/(kg*K)
        "R_v": 461.9, # J/(kg*K)
    },
    "hab2": {
        # "albedo": 0.3, ?
        "R_d": 189.0, # J/(kg*K) mu_d = 44.0 g/mol (CO2)
        "R_v": 461.9, # J/(kg*K)
    }
}

THAI_PARAMS = {
    "exocam_lon_substellar": 180.0,
    "lmdg_lon_substellar": 0.0,
    "rock3d_lon_substellar": 180.0,
    "um_lon_substellar": 0.0,
}

