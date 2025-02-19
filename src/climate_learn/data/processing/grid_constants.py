import numpy as np
P_LEVELS = np.logspace(1, 5, 10) # log-spaced pressure levels from 10 Pa to 10^5 Pa
LAT, LON = np.linspace(-90, 90, 32), np.linspace(0, 360, 64, endpoint=False) # they seem to use 0, 360 with endpoint=False