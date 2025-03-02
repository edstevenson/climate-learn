import numpy as np
P_LEVELS = np.array([92_500, 80_000, 62_500, 40_000, 25_000, 15_000, 6000, 2000, 500, 20]) 
LAT, LON = np.linspace(-90, 90, 32), np.linspace(-180, 180, 64, endpoint=False) # they seem to use 0, 360 with endpoint=False 