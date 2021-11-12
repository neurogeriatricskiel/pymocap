import numpy as np
import pandas as pd
import os
from pymocap import preprocessing
from pymocap.algo.predict_missing_markers import predict_missing_markers
from pymocap.algo import *
from pymocap.utils import _load_file
import matplotlib.pyplot as plt

def split_train_val_test_ids(file_name, seed=None):
    """Splits the participant ids into three separate groups:
    one for training, one for validating and one for testing, respectively.

    Parameters
    ----------
    file_name : str
        The name of file where ids are linked to a specific participant group.
    seed : int, optional
        Seed for random number generator, by default None
    
    Returns
    -------
    train_ids, val_ids, test_ids : list, list, list
        A list of participant ids for the different sets.
    """
    np.random.seed(seed) if seed is not None else np.random.seed(123)

    # Initialize output lists
    train_ids, val_ids, test_ids = [], [], []

    # Load file into pandas DataFrame
    df = pd.read_csv(file_name, sep=",", header=0)

    # Loop over the participant types
    for participant_type in df["participant_type"].unique():

        # Get participant ids
        participant_ids = df[df["participant_type"]==participant_type]["participant_id"].values.tolist()

        # Get number of participants
        n_participants = len(participant_ids)

        # Calculate number of ids to go into the different set
        n_train_ids = n_participants // 2
        n_val_ids   = (n_participants - n_train_ids) // 2
        n_test_ids  = n_participants - n_train_ids - n_val_ids

        # Get random ids
        ix_random = np.random.choice(participant_ids, n_train_ids+n_val_ids, replace=False)
        
        # Add to corresponding lists
        train_ids += [ix for ix in ix_random[:n_train_ids]]
        val_ids   += [ix for ix in ix_random[n_train_ids:]]
        test_ids  += [ix for ix in participant_ids if (ix not in train_ids) and (ix not in val_ids)]

    return train_ids, val_ids, test_ids

def get_start_and_end_index(start_1, start_2, end_1, end_2, l_psis, r_psis):    
    """Determines the start and end index of the trial.
    The start is defined as the instant where the right or left psis marker crosses the start line.
    The end is defined as the instant that the left or right psis marker crosses the end line.

    Parameters
    ----------
    start_1 : array_like
        Marker position data for the respective markers.
    start_2 : [type]
        [description]
    end_1 : [type]
        [description]
    end_2 : [type]
        [description]
    l_psis : [type]
        [description]
    r_psis : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    # Midpoint between two start and two end markers
    mid_start = ( start_1 + start_2 ) / 2
    mid_end   = ( end_1 + end_2 ) / 2

    # Distance from relevant markers to midstart
    dist_start_l_psis = np.sqrt(np.sum((l_psis[:,:2] - mid_start[:2])**2, axis=1))
    dist_start_r_psis = np.sqrt(np.sum((r_psis[:,:2] - mid_start[:2])**2, axis=1))
    dist_end_l_psis = np.sqrt(np.sum((l_psis[:,:2] - mid_end[:2])**2, axis=1))
    dist_end_r_psis = np.sqrt(np.sum((r_psis[:,:2] - mid_end[:2])**2, axis=1))

    # Time index
    ix_start = np.min((np.argmin(dist_start_l_psis), np.argmin(dist_start_r_psis)))
    ix_start = np.max((0, ix_start))
    ix_end   = np.nanmin((np.argmin(dist_end_l_psis), np.argmin(dist_end_r_psis), l_psis.shape[0]))

    # Sanity check, if crossing the end before the start, then swap the marker labels
    if ix_end < ix_start:
        ix_start_ = ix_end
        ix_end = ix_start
        ix_start = ix_start_
        del ix_start_
    return ix_start, ix_end

def create_sequences(input_data, seq_len, step_len):
    """Segments the input data into sequences of equal length.

    Parameters
    ----------
    input_data : (N, D) array_like
        Input data with N discrete time step and D input channels.
    seq_len : int
        Number of samples for any single segment.
    step_len : int
        Number of samples to move forward.
    """
    if input_data.shape[0] < seq_len:
        return
    if step_len == 0:
        step_len = seq_len
    
    # Loop over the data
    output_data = []
    for i in range(0, input_data.shape[0]-seq_len+1, step_len):
        output_data.append(input_data[i:i+seq_len,:])
    return np.stack(output_data)

def load_dataset(file_name, dir_name, ids, set_name="training"):
    # Load file into pandas DataFrame
    df = pd.read_csv(file_name, sep=",", header=0)

    # Trials
    trial_names = ["walk_fast", "walk_preferred", "walk_slow"]

    # For each participant id
    for participant_id in ids:

        # For each trial
        for trial_name in trial_names:

            # If trial for participant id is included for analysis
            if df[df["participant_id"]==participant_id][trial_name].values == 1:
                
                # --------------------------------------------------
                # Optical motion capture data
                # --------------------------------------------------

                # Load data
                try:
                    omc = _load_file(os.path.join(dir_name, participant_id, "optical", "omc_"+trial_name+".mat"))
                except:
                    omc = _load_file(os.path.join(dir_name, participant_id, "optical", "omc_"+trial_name+"_on.mat"))
                
                # Get indexes for relevant markers
                ix_l_heel = omc['marker_location'].tolist().index('l_heel')
                ix_l_toe  = omc['marker_location'].tolist().index('l_toe')
                ix_l_psis = omc['marker_location'].tolist().index('l_psis')
                ix_r_heel = omc['marker_location'].tolist().index('r_heel')
                ix_r_toe  = omc['marker_location'].tolist().index('r_toe')
                ix_r_psis = omc['marker_location'].tolist().index('r_psis')
                start_1 = np.nanmean(omc['pos'][:,:3,omc['marker_location'].tolist().index('start_1')], axis=0)
                start_2 = np.nanmean(omc['pos'][:,:3,omc['marker_location'].tolist().index('start_2')], axis=0)
                end_1   = np.nanmean(omc['pos'][:,:3,omc['marker_location'].tolist().index('end_1')], axis=0)
                end_2   = np.nanmean(omc['pos'][:,:3,omc['marker_location'].tolist().index('end_2')], axis=0)

                # Reshape data
                pos = np.reshape(omc['pos'][:,:3,:], (omc['pos'].shape[0], (omc['pos'].shape[1]-1)*omc['pos'].shape[2]), order='F')

                # Interpolate data
                filled_pos = predict_missing_markers(pos)

                # Get indexes for start and end of trial
                ix_start, ix_end = get_start_and_end_index(start_1, start_2, end_1, end_2, filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_r_psis*3:ix_r_psis*3+3])

                # Detect gait events
                ix_l_IC_OConnor, ix_l_FC_OConnor = oconnor(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], omc['fs'])
                ix_r_IC_OConnor, ix_r_FC_OConnor = oconnor(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], omc['fs'])

                ix_l_IC_Pijnappels, ix_l_FC_Pijnappels = pijnappels(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], omc['fs'])
                ix_r_IC_Pijnappels, ix_r_FC_Pijnappels = pijnappels(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], omc['fs'])

                ix_l_IC_Zeni, ix_l_FC_Zeni = zeni(filled_pos[:,ix_l_heel*3:ix_l_heel*3+3], filled_pos[:,ix_l_toe*3:ix_l_toe*3+3], filled_pos[:,ix_l_psis*3:ix_l_psis*3+3], omc['fs'])
                ix_r_IC_Zeni, ix_r_FC_Zeni = zeni(filled_pos[:,ix_r_heel*3:ix_r_heel*3+3], filled_pos[:,ix_r_toe*3:ix_r_toe*3+3], filled_pos[:,ix_r_psis*3:ix_r_psis*3+3], omc['fs'])
                    

                # --------------------------------------------------
                # Inertial measurement units data
                # --------------------------------------------------

                # Load data
                try:
                    imu = _load_file(os.path.join(dir_name, participant_id, "imu", "imu_"+trial_name+".mat"))
                except:
                    imu = _load_file(os.path.join(dir_name, participant_id, "imu", "imu_"+trial_name+"_on.mat"))
                
                # Get indexes for relevant sensors
                ix_left_ankle = imu['imu_location'].tolist().index('left_ankle')
                ix_right_ankle = imu['imu_location'].tolist().index('right_ankle')
                
                # Reshape data
                acc = np.reshape(imu['acc'], (imu['acc'].shape[0], imu['acc'].shape[1]*imu['acc'].shape[2]), order='F')
                gyro= np.reshape(imu['gyro'], (imu['gyro'].shape[0], imu['gyro'].shape[1]*imu['gyro'].shape[2]), order='F')
                magn= np.reshape(imu['magn'], (imu['magn'].shape[0], imu['magn'].shape[1]*imu['magn'].shape[2]), order='F')

                # If necessary, then resample sensor data
                if imu['fs'] != omc['fs']:
                    acc  = preprocessing._resample_data(acc, imu['fs'], omc['fs'])
                    gyro = preprocessing._resample_data(gyro, imu['fs'], omc['fs'])
                    magn = preprocessing._resample_data(magn, imu['fs'], omc['fs'])
                
                # Detect gait events
                ix_l_IC_Salarian, ix_l_FC_Salarian = salarian(-gyro[:,ix_left_ankle*3+2], imu['fs'])
                ix_r_IC_Salarian, ix_r_FC_Salarian = salarian(gyro[:,ix_right_ankle*3+2], imu['fs'])

                # Visualize
                visualize = True
                if visualize:
                    fig, axs = plt.subplots(4, 1, figsize=(21., 14.8))
                    axs[0].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*np.nanmax(pos[:,ix_l_heel*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[0].fill_between(np.arange(ix_end, pos.shape[0]), np.ones((pos.shape[0]-ix_end,))*np.nanmax(pos[:,ix_l_heel*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[1].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*np.nanmax(pos[:,ix_l_toe*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[1].fill_between(np.arange(ix_end, pos.shape[0]), np.ones((pos.shape[0]-ix_end,))*np.nanmax(pos[:,ix_l_toe*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[2].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*np.nanmax(pos[:,ix_r_heel*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[2].fill_between(np.arange(ix_end, pos.shape[0]), np.ones((pos.shape[0]-ix_end,))*np.nanmax(pos[:,ix_r_heel*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[3].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*np.nanmax(pos[:,ix_r_toe*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[3].fill_between(np.arange(ix_end, pos.shape[0]), np.ones((pos.shape[0]-ix_end,))*np.nanmax(pos[:,ix_r_toe*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')

                    axs[0].plot(pos[:,ix_l_heel*3+2], '-', c=(0, 0, 0, 0.2), lw=1)
                    for i in range(len(ix_l_IC_OConnor)):
                        axs[0].plot([ix_l_IC_OConnor[i], ix_l_IC_OConnor[i]], [np.nanmin(pos[:,ix_l_heel*3+2]), np.nanmax(pos[:,ix_l_heel*3+2])], '-', c=(0, 0, 1), lw=1)
                    axs[0].set_xlim((0, pos.shape[0]))

                    axs[1].plot(pos[:,ix_l_toe*3+2], '-', c=(0, 0, 0, 0.2), lw=1)
                    for i in range(len(ix_l_FC_Zeni)):
                        axs[1].plot([ix_l_FC_Zeni[i], ix_l_FC_Zeni[i]], [np.nanmin(pos[:,ix_l_toe*3+2]), np.nanmax(pos[:,ix_l_toe*3+2])], '-', c=(0, 0, 1), lw=1)
                    axs[1].set_xlim((0, pos.shape[0]))

                    axs[2].plot(pos[:,ix_r_heel*3+2], '-', c=(0, 0, 0, 0.2), lw=1)
                    for i in range(len(ix_r_IC_OConnor)):
                        axs[2].plot([ix_r_IC_OConnor[i], ix_r_IC_OConnor[i]], [np.nanmin(pos[:,ix_r_heel*3+2]), np.nanmax(pos[:,ix_r_heel*3+2])], '-', c=(0, 0, 1), lw=1)
                    axs[2].set_xlim((0, pos.shape[0]))

                    axs[3].plot(pos[:,ix_r_toe*3+2], '-', c=(0, 0, 0, 0.2), lw=1)
                    for i in range(len(ix_r_FC_Zeni)):
                        axs[3].plot([ix_r_FC_Zeni[i], ix_r_FC_Zeni[i]], [np.nanmin(pos[:,ix_r_toe*3+2]), np.nanmax(pos[:,ix_r_toe*3+2])], '-', c=(0, 0, 1), lw=1)
                    axs[3].set_xlim((0, pos.shape[0]))

                    # IMU
                    for i in range(len(ix_l_IC_Salarian)):
                        axs[0].plot([ix_l_IC_Salarian[i], ix_l_IC_Salarian[i]], [np.nanmin(pos[:,ix_l_heel*3+2]), np.nanmax(pos[:,ix_l_heel*3+2])], '-', c=(0, 0.5, 0), lw=1)
                    for i in range(len(ix_l_FC_Salarian)):
                        axs[1].plot([ix_l_FC_Salarian[i], ix_l_FC_Salarian[i]], [np.nanmin(pos[:,ix_l_toe*3+2]), np.nanmax(pos[:,ix_l_toe*3+2])], '-', c=(0, 0.5, 0), lw=1)
                    for i in range(len(ix_r_IC_Salarian)):
                        axs[2].plot([ix_r_IC_Salarian[i], ix_r_IC_Salarian[i]], [np.nanmin(pos[:,ix_r_heel*3+2]), np.nanmax(pos[:,ix_r_heel*3+2])], '-', c=(0, 0.5, 0), lw=1)
                    for i in range(len(ix_r_FC_Salarian)):
                        axs[3].plot([ix_r_FC_Salarian[i], ix_r_FC_Salarian[i]], [np.nanmin(pos[:,ix_r_toe*3+2]), np.nanmax(pos[:,ix_r_toe*3+2])], '-', c=(0, 0.5, 0), lw=1)
                    plt.show()

                # Prepare data array
                data = np.hstack((acc[:,ix_left_ankle*3:ix_left_ankle*3+3],    # accelerometer left
                                gyro[:,ix_left_ankle*3:ix_left_ankle*3+3],     # gyroscope left
                                magn[:,ix_left_ankle*3:ix_left_ankle*3+3],     # magnetometer left
                                acc[:,ix_right_ankle*3:ix_right_ankle*3+3],    # accelerometer right
                                gyro[:,ix_right_ankle*3:ix_right_ankle*3+3],   # gyroscope right
                                magn[:,ix_right_ankle*3:ix_right_ankle*3+3]))  # magnetometer right
                
                # Prepare labels array
                labels = np.zeros((data.shape[0], 4))
                labels[ix_l_IC_Pijnappels,0] = 1  # left initial contact
                labels[ix_l_FC_Zeni,1] = 1     # left final contact
                labels[ix_r_IC_Pijnappels,2] = 1  # right initial contact
                labels[ix_r_FC_Zeni,3] = 1     # right final contact

                # Create sequences of equal sequence length, for input to the neural network
                if set_name == "training":
                    data = create_sequences(np.hstack((data, labels)), seq_len=300, step_len=50)

                # Save to output file
                output_file_name = participant_id + "_" + trial_name
                with open(os.path.join("/home/robr/Desktop", output_file_name), 'wb') as outfile:
                    np.save(outfile, data)


if __name__ == "__main__":
    CSV_FILE_NAME = os.path.join('/mnt/neurogeriatrics_data/Keep Control/Output/lab dataset', 'participants.csv')
    ROOT_DIR = '/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset'
    
    train_ids, val_ids, test_ids = split_train_val_test_ids(CSV_FILE_NAME)
    load_dataset(file_name=CSV_FILE_NAME, dir_name=ROOT_DIR, ids=train_ids, set_name="training")