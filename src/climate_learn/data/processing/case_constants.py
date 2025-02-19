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
        "R_d": 296.95, # J/(kg*K) TODO: these are placeholder values -- can correct from input file when found (so are the other cases below)
        # placeholder calcs:
        # R_d = R_univ / mu_d; R_univ = 8.3145 J/(K*mol), mu_d = 28.0 g/mol
        # R_v = R_univ / mu_v; mu_v = 18.0 g/mol    
        "R_v": 461.92, # J/(kg*K) 
    },
    "ben2": {
        "albedo": 0.3,
        "R_d": 296.95, # J/(kg*K)
        "R_v": 461.92, # J/(kg*K)
    }
}


