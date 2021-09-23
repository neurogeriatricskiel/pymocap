# Load libraries
from lib.utils import _load_file
from lib.preprocessing import _predict_missing_markers
import numpy as np
import os, re

def mean_euclidean_distance(y, yhat):
    """Calculates the mean Euclidean distance between the original and reconstruct marker trajectory.

    Parameters
    ----------
    y : (N, 3) array_like)
        The original marker trajectory for N time steps of missing data.
    yhat : (N, 3) array_like
        The reconstructed marker trajectory.
    
    Returns
    -------
    _ : float
        The mean Euclidean distance.
    """

    # Compute the element-wise differences
    differences = ( yhat - y )

    # Take the squared values
    squared_differences = differences**2

    # Sum for each time step
    sum_squared_differences = np.sum(squared_differences, axis=1)

    # Take the square root
    distances = np.sqrt(sum_squared_differences)
    return np.mean(distances)

# Set data directory
PARENT_FOLDER = "/home/robr/Code/pymocap/data/test"

# Load original data
data_full = _load_file(os.path.join(PARENT_FOLDER, "WalkL.mat"))

# Get number of time steps, number of markers
n_time_steps, n_channels = data_full.shape
n_markers = n_channels // 3

# Get a list of filenames
filenames = [fname for fname in os.listdir(PARENT_FOLDER) if re.match("WalkL_[0-9]{4}.mat", fname)]

# Iterate of the files
scores = {}
for mrk in range(n_markers):
    scores[mrk] = {"files": [], "scores": [], "gap_length": []}

for (i, filename) in enumerate(filenames[:3]):
    print(filename)

    # Load data from file
    data_gaps = _load_file(os.path.join(PARENT_FOLDER, filename))
    
    # Call method to reconstruct data
    data_filled = _predict_missing_markers(data_gaps, method="R2")

    # Find channels with missing data
    ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=0))
    ix_time_steps_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=1))
    ix_markers_with_gaps = ( ix_channels_with_gaps[2::3] // 3 )
    
    # Iterate over the markers with gaps
    for mrk in ix_markers_with_gaps:

        # Find which time steps are missing
        ix_time_steps_with_gaps_mrk, = np.nonzero(np.any(np.isnan(data_gaps[:,mrk*3:mrk*3+3]), axis=1))
        
        # Compute the mean Euclidean distance for the current marker
        dist = mean_euclidean_distance(data_full[ix_time_steps_with_gaps_mrk,mrk*3:mrk*3+3], data_filled[ix_time_steps_with_gaps_mrk,mrk*3:mrk*3+3])

        # Add infos to dictionary
        scores[mrk]["files"].append(filename)
        scores[mrk]["scores"].append(dist)
        scores[mrk]["gap_length"].append(len(ix_time_steps_with_gaps_mrk))
print(scores)