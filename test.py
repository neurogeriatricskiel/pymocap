# Load libraries
from lib.utils import _load_file
from lib.preprocessing import _predict_missing_markers
import numpy as np
import os, re

# Set data directory
PARENT_FOLDER = "/home/robr/Code/pymocap/data/test"

# Get a list of filenames
filenames = [fname for fname in os.listdir(PARENT_FOLDER) if re.match("WalkL_[0-9]{4}.mat", fname)]

# Iterate of the files
for (i, filename) in enumerate(filenames):
    print(filename)

    # Load data from file
    data_gaps = _load_file(os.path.join(PARENT_FOLDER, filename))
    
    # Call method to reconstruct data
    data_filled = _predict_missing_markers(data_gaps, method="R1")