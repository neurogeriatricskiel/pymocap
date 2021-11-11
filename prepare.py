import os, sys
import numpy as np
import pandas as pd
from pymocap.utils import _load_file
from pymocap.algo import *

def split_train_val_test_ids():
    """Splits the dataset into a set for training, validating and test.

    Returns
    -------
    train_ids, val_ids, test_ids : list-like, list-like, list-like
        A list-like with the participant ids for the corresponding datasets.
    """
    
    # Set seed for reproducible results
    np.random.seed(123)

    # Initialize list-like for ids
    train_ids, val_ids, test_ids = [], [], []

    # For each participant type
    for participant_type in df["participant_type"].unique():

        # Get corresponding participant ids
        participant_ids = df[df["participant_type"]==participant_type]["participant_id"].values.tolist()
                
        # Put half of the number of participants in training set
        n_participants = len(participant_ids)//2
        cnt = 0
        while cnt < n_participants:
            ix_random = np.random.choice(len(participant_ids), 1, replace=False)[0]
            train_ids.append(participant_ids[ix_random])
            participant_ids.remove(participant_ids[ix_random])
            cnt += 1
        
        # From the remaining participants, put half of them in test set
        n_participants = len(participant_ids)//2 if (len(participant_ids)//2)>0 else 1
        cnt = 0
        while cnt < n_participants:
            ix_random = np.random.choice(len(participant_ids), 1, replace=False)[0]
            test_ids.append(participant_ids[ix_random])
            participant_ids.remove(participant_ids[ix_random])
            cnt += 1
        
        # The remaining ids are used for validation
        val_ids += participant_ids
    return train_ids, val_ids, test_ids
    
def _make_dataset(path, type="train"):

    # Get list of train ids
    if type == "train":
        participant_ids = train_ids
    elif type == "val":
        participant_ids = val_ids
    else:
        participant_ids = test_ids
    
    for i, participant_id in enumerate(participant_ids):
        trial_names = []
        if df[df["participant_id"]==participant_id]["walk_fast"].values == 1:
            trial_names.append("walk_fast")
        if df[df["participant_id"]==participant_id]["walk_preferred"].values == 1:
            trial_names.append("walk_preferred")
        if df[df["participant_id"]==participant_id]["walk_slow"].values == 1:
            trial_names.append("walk_slow")

        # ------------------------- OPTICAL MOTION CAPTURE -------------------------
        # Get optical motion capture data
        file_names = [file_name for file_name in os.listdir(os.path.join(path, participant_id, "optical")) if file_name.endswith(".mat")]
        for trial_name in trial_names:
            ix_file_name = [ix for ix in range(len(file_names)) if trial_name in file_names[ix]][0]
            print(file_names[ix_file_name])
            omc = _load_file(os.path.join(input_data_dir, participant_id, "optical", file_names[ix_file_name]))

            # Check if data is available for relevant markers
            my_markers = ["l_heel", "r_heel", "l_toe", "r_toe", "l_psis", "r_psis"]
            for marker in my_markers:
                if (marker not in omc["marker_location"]):
                    print(f"Data from `{marker:s}` not available from {file_names[ix_file_name]:s} of {participant_id:s}!")

            # Indexes corresponding to relevant heel and toe markers
            ix_l_heel = np.argwhere(omc["marker_location"]=="l_heel")[:,0][0]
            ix_r_heel = np.argwhere(omc["marker_location"]=="r_heel")[:,0][0]
            ix_l_toe = np.argwhere(omc["marker_location"]=="l_toe")[:,0][0]
            ix_r_toe = np.argwhere(omc["marker_location"]=="r_toe")[:,0][0]
            ix_l_psis = np.argwhere(omc["marker_location"]=="l_psis")[:,0][0]
            ix_r_psis = np.argwhere(omc["marker_location"]=="r_psis")[:,0][0]
            start_1 = np.nanmean(omc["pos"][:,:3,np.argwhere(omc["marker_location"]=="start_1")[:,0][0]], axis=0)
            start_2 = np.nanmean(omc["pos"][:,:3,np.argwhere(omc["marker_location"]=="start_2")[:,0][0]], axis=0)
            end_1 = np.nanmean(omc["pos"][:,:3,np.argwhere(omc["marker_location"]=="end_1")[:,0][0]], axis=0)
            end_2 = np.nanmean(omc["pos"][:,:3,np.argwhere(omc["marker_location"]=="end_2")[:,0][0]], axis=0)

            # Reshape data
            pos = np.reshape(omc["pos"][:,:3,:], (omc["pos"].shape[0], (omc["pos"].shape[1]-1)*omc["pos"].shape[2]), order='F')

            # Fill gaps in marker data
            try:
                filled_pos = predict_missing_markers(pos)
            except:
                continue

            # Detect ICs and FCs
            ix_l_IC_OConnor, ix_l_FC_OConnor = oconnor(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], omc_data['fs'])
            ix_r_IC_OConnor, ix_r_FC_OConnor = oconnor(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], omc_data['fs'])

            ix_l_IC_Pijnappels, ix_l_FC_Pijnappels = pijnappels(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], omc_data['fs'])
            ix_r_IC_Pijnappels, ix_r_FC_Pijnappels = pijnappels(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], omc_data['fs'])

            ix_l_IC_Zeni, ix_l_FC_Zeni = zeni(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], filled_pos[:,ix_l_psis*3:ix_l_psis*3+3], omc_data['fs'])
            ix_r_IC_Zeni, ix_r_FC_Zeni = zeni(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], filled_pos[:,ix_r_psis*3:ix_r_psis*3+3], omc_data['fs'])

        break
    return

if __name__ == "__main__":

    # Load demographics data into pandas DataFrame
    df = pd.read_csv(os.path.join("Z:\\Keep Control\\Output\\lab dataset\\", "participants.csv"), sep=",", header=0)

    # Split participant ids into set for training, validating, and testing
    train_ids, val_ids, test_ids = split_train_val_test_ids()
    
    # Prepare the data for each dataset
    input_data_dir = "Z:\\Keep Control\\Data\\lab dataset"
    _make_dataset(input_data_dir)
    