from pymocap import preprocessing
import numpy as np
import matplotlib.pyplot as plt

# Generate data
fs = 100.0
N  = 1000
t  = np.arange(N)/fs
x  = 3 * np.cos(2*np.pi*3*t)
y  = np.hstack((x.reshape(-1,1), x.reshape(-1,1), x.reshape(-1,1)))

# Introduce some NaN values
for j in range(y.shape[-1]):
    for i in np.random.choice(N, 3, replace=False):
        y[i:i+10,j] = np.nan

# Resample data to interpolate
z = preprocessing._resample_data(y, fs, fs)

# Plot figure
fig, ax = plt.subplots(1, 1)
ax.plot(t, x, ls=':', lw=2, c=(0, 0, 0))
ax.plot(t, y, ls='-', lw=2, c=(0, 0, 0))
ax.plot(t, z, ls='-', lw=1, c=(1, 0, 0), marker='*', mfc='none', mec='r', ms=4)
plt.show()