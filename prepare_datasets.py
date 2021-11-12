import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymocap.utils import _load_file

def main():
    if sys.platform == "linux":
        # Read participants information
        df = pd.read_csv(os.path.join("/mnt/neurogeriatrics_data/Keep Control/Output/lab dataset", "participants.csv"), sep=",", header=0)
    
    # Groups info
    ds = {}
    for participant_type in df["participant_type"].unique():
        ds[participant_type] = {"train": [], "val": [], "test": []}
        n_train_ids = len(df[df["participant_type"]==participant_type])//2
        n_val_ids = (len(df[df["participant_type"]==participant_type])-n_train_ids)//2
        n_test_ids = len(df[df["participant_type"]==participant_type]) - n_train_ids - n_val_ids
        print(f"{participant_type}:")
        print(f"  Number of participant ids: {len(df[df['participant_type']==participant_type])}")
        print(f"    Number of training ids: {n_train_ids}")
        print(f"    Number of validation ids: {n_val_ids}")
        print(f"    Number of testing ids: {n_test_ids}")
    return ds

if __name__ == "__main__":
    # Run main()
    ds = main()