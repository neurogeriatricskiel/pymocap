from numpy.lib.index_tricks import ix_
from pymocap.algo.oconnor import oconnor


def split_train_val_test_files(input_data_dir, file_name, ddata, seed=None):
    """Splits dataset into a training, validation and test set.
    
    Participants are randomly assigned to a set, stratified by participant type.
    
    Parameters
    ----------
    input_data_dir : str
        The relative or absolute path to the directory where the data is stored.
    file_name : str
        The filename of the .csv file that contains the demographics data.
    ddata : str
        The filename of the .json data file, with per participant id which files to take into account.
    seed : int
        A seed for the random number generator.
    
    Returns
    -------
    ... : ...
    """
    import pandas as pd
    import numpy as np
    import json
    
    def _populate_with_ids(participant_ids, n_samples, seed=seed):
        """Populates sets with randomly selected participant ids.
        
        Parameters
        ----------
        participant_ids : list
            A list of participant ids.
        n_samples : int
            The number of ids that need to be sampled into the set.
        
        Returns
        -------
        selected_ids, remaining_ids : list
            A list of selected participant ids that go into the training, validation or test set, and
            the list of remaining ids, respectively.
        """
        np.random.seed(123) if seed is None else np.random.seed(seed)  # for reproducibility
        
        # Initialize counter
        cnt = 0
        
        # Randomly sample participant ids for the current set
        # until the number of samples is reached
        selected_ids = []
        while cnt < n_samples:
            
            # Sample a random participant ids
            random_participant_id = np.random.choice(participant_ids, 1, replace=False)[0]
            
            # Add to list
            selected_ids.append(random_participant_id)
            
            # Remove from participant ids
            participant_ids.remove(random_participant_id)
            
            # Increment counter
            cnt += 1
        return selected_ids, participant_ids
    
    def _find_file(participant_id, trial_name):
        """Find the filename for a given participant and walking trial.
        
        Parameters
        ----------
        participant_id : str
            The participant id, in the form `pp###`
        trial_name : str
            The name of the walking trial, e.g., `walk_slow`, ...
        
        Returns
        -------
        _ : str
            The exact filename, for the optical motion capture data file, e.g., omc_walk_slow.mat`.
        """
        import os
        
        filenames = [filename for filename in os.listdir(os.path.join(input_data_dir, participant_id, "optical")) if filename.endswith(".mat")]
        ix_filename = [ix for ix in range(len(filenames)) if trial_name in filenames[ix]]
        if len(ix_filename) > 0:
            return filenames[ix_filename[0]]
        
    # Load data into pandas DataFrame    
    df = pd.read_csv(file_name, sep=",", header=0)
    
    # Loop over the participant types
    train_ids, val_ids, test_ids = [], [], []
    for participant_type in df["participant_type"].unique():
        
        # Get number of participants for current type
        n_participants = len(df[df["participant_type"]==participant_type])
        
        # Set number of training examples
        n_train = n_participants // 2

        # Set number of validation examples
        n_val = ( n_participants - n_train ) // 2

        # Set number of test examples
        n_test = ( n_participants - n_train - n_val )
        
        # Get current participant ids
        current_participant_ids = df[df["participant_type"]==participant_type]["participant_id"].values.tolist()
        
        # Assign ids for training set
        train_ids_, current_participant_ids = _populate_with_ids(current_participant_ids, n_train)
        val_ids_, test_ids_ = _populate_with_ids(current_participant_ids, n_val)
        
        # Accumulate ids over consecutive participant types
        train_ids += train_ids_
        val_ids += val_ids_
        test_ids += test_ids_
    
    # Read data from json file
    with open(ddata, 'rb') as json_file:
        json_info = json.load(json_file)
    
    # Create dictionaries for training, validation and test set
    train_files, val_files, test_files = {}, {}, {}
    for train_id in train_ids:
        train_files[train_id] = []
        for trial_name in json_info[train_id]:
            train_files[train_id].append(_find_file(train_id, trial_name))
            
    for val_id in val_ids:
        val_files[val_id] = []
        for trial_name in json_info[val_id]:
            val_files[val_id].append(_find_file(val_id, trial_name))
    
    for test_id in test_ids:
        test_files[test_id] = []
        for trial_name in json_info[test_id]:
            test_files[test_id].append(_find_file(test_id, trial_name))
            
    return train_files, val_files, test_files

def get_labels(omc_data, **kwargs):
    """Gets the labels, i.e. gait events, from the optical motion capture data.

    Parameters
    ----------
    omc_data : dictionary
        A nested dictionary with the following keys:
            fs : int, float
                The sampling frequency (in Hz).
            pos : (N, 4, M) array_like
                The raw marker position data (in mm), for N discrete time steps and M markers.
            marker_location : (M,) array_like
                The specification of the marker locations.
    
    Returns
    -------
    ix_l_IC, ix_l_FC, ix_r_IC, ix_r_FC : array_like
        Array with indexes corresponding to initial and final contacts.
    ix_start, ix_end : array_like
        Index corresponding to start and end of trial.
    """
    import numpy as np
    from pymocap.algo.predict_missing_markers import predict_missing_markers
    from pymocap.algo import pijnappels, zeni
    import matplotlib.pyplot as plt

    visualize = kwargs.get("visualize", False)

    # Get indexes of relevant
    ix_l_heel = omc_data["marker_location"].tolist().index('l_heel')
    ix_r_heel = omc_data["marker_location"].tolist().index('r_heel')
    ix_l_toe  = omc_data["marker_location"].tolist().index('l_toe')
    ix_r_toe  = omc_data["marker_location"].tolist().index('r_toe')
    ix_l_psis = omc_data["marker_location"].tolist().index('l_psis')
    ix_r_psis = omc_data["marker_location"].tolist().index('r_psis')

    # Reshape marker position data
    pos = np.reshape(omc_data["pos"][:,:3,:], (omc_data["pos"].shape[0], (omc_data["pos"].shape[1]-1)*omc_data["pos"].shape[2]), order='F')

    # Fill any possible gaps in the marker trajectories
    pos = predict_missing_markers(pos)

    # Get start and end of trial
    start_1 = np.nanmean(omc_data["pos"][:,:3,omc_data["marker_location"].tolist().index('start_1')], axis=0)
    start_2 = np.nanmean(omc_data["pos"][:,:3,omc_data["marker_location"].tolist().index('start_2')], axis=0)
    end_1 = np.nanmean(omc_data["pos"][:,:3,omc_data["marker_location"].tolist().index('end_1')], axis=0)
    end_2 = np.nanmean(omc_data["pos"][:,:3,omc_data["marker_location"].tolist().index('end_2')], axis=0)
    mid_start = ( start_1 + start_2 ) / 2
    mid_end = ( end_1 + end_2 ) / 2

    # Distance from relevant markers to midstart
    dist_start_l_psis = np.sqrt(np.sum((pos[:,ix_l_psis*3:ix_l_psis*3+2] - mid_start[:2])**2, axis=1))
    dist_start_r_psis = np.sqrt(np.sum((pos[:,ix_r_psis*3:ix_r_psis*3+2] - mid_start[:2])**2, axis=1))
    dist_end_l_psis = np.sqrt(np.sum((pos[:,ix_l_psis*3:ix_l_psis*3+2] - mid_end[:2])**2, axis=1))
    dist_end_r_psis = np.sqrt(np.sum((pos[:,ix_r_psis*3:ix_r_psis*3+2] - mid_end[:2])**2, axis=1))

    # Time index
    ix_start = np.min((np.argmin(dist_start_l_psis), np.argmin(dist_start_r_psis)))
    ix_start = np.max((0, ix_start))
    ix_end   = np.nanmin((np.argmin(dist_end_l_psis), np.argmin(dist_end_r_psis), pos.shape[0]))

    # Sanity check, if crossing the end before the start, then swap the marker labels
    if ix_end < ix_start:
        ix_start_ = ix_end
        ix_end = ix_start
        ix_start = ix_start_
        del ix_start_

    # Detect initial contacts and final contacts
    ix_l_IC_Pijnappels, _ = pijnappels(pos[:,ix_l_heel*3:ix_l_heel*3+3], pos[:,ix_l_toe*3:ix_l_toe*3+3], omc_data['fs'])
    _, ix_l_FC_Zeni = zeni(pos[:,ix_l_heel*3:ix_l_heel*3+3], pos[:,ix_l_toe*3:ix_l_toe*3+3], pos[:,ix_l_psis*3:ix_l_psis*3+3], omc_data['fs'])
    ix_r_IC_Pijnappels, _ = pijnappels(pos[:,ix_r_heel*3:ix_r_heel*3+3], pos[:,ix_r_toe*3:ix_r_toe*3+3], omc_data['fs'])
    _, ix_r_FC_Zeni = zeni(pos[:,ix_r_heel*3:ix_r_heel*3+3], pos[:,ix_r_toe*3:ix_r_toe*3+3], pos[:,ix_r_psis*3:ix_r_psis*3+3], omc_data['fs'])
    
    if visualize == True:
        mi = np.min([np.nanmin(omc_data["pos"][:,2,ix_l_heel]), np.nanmin(omc_data["pos"][:,2,ix_l_toe]), np.nanmin(omc_data["pos"][:,2,ix_r_heel]), np.nanmin(omc_data["pos"][:,2,ix_r_toe])])
        ma = np.max([np.nanmax(omc_data["pos"][:,2,ix_l_heel]), np.nanmax(omc_data["pos"][:,2,ix_l_toe]), np.nanmax(omc_data["pos"][:,2,ix_r_heel]), np.nanmax(omc_data["pos"][:,2,ix_r_toe])])

        fig, ax = plt.subplots(1, 1, figsize=(21, 14.8))
        # axs[0][0].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*np.max(filtered_pos[:,ix_l_heel*3+2]), color=(0, 0, 0), alpha=0.1, ec='none')
        ax.fill_between(np.arange(0, ix_start), np.ones((ix_start,))*ma, color=(0, 0, 0), alpha=0.05, ec='none')
        ax.fill_between(np.arange(0, ix_start), np.ones((ix_start,))*mi, color=(0, 0, 0), alpha=0.05, ec='none')
        ax.fill_between(np.arange(ix_end, omc_data["pos"].shape[0]), np.ones((omc_data["pos"].shape[0]-ix_end,))*ma, color=(0, 0, 0), alpha=0.05, ec='none')
        ax.fill_between(np.arange(ix_end, omc_data["pos"].shape[0]), np.ones((omc_data["pos"].shape[0]-ix_end,))*mi, color=(0, 0, 0), alpha=0.05, ec='none')

        ax.plot(omc_data["pos"][:,2,ix_l_heel], '-', c=(0, 0, 1, 0.3))
        ax.plot(omc_data["pos"][:,2,ix_l_toe], ':', c=(0, 0, 1, 0.3))
        ax.plot(omc_data["pos"][:,2,ix_r_heel], '-', c=(1, 0, 0, 0.3))
        ax.plot(omc_data["pos"][:,2,ix_r_toe], ':', c=(1, 0, 0, 0.3))

        ax.plot(ix_l_IC_Pijnappels, omc_data["pos"][ix_l_IC_Pijnappels,2,omc_data["marker_location"].tolist().index('l_heel')], 'x', c=(0, 0, 1))
        ax.plot(ix_l_FC_Zeni, omc_data["pos"][ix_l_FC_Zeni,2,omc_data["marker_location"].tolist().index('l_toe')], 'o', mfc='none', mec=(0, 0, 1))
        ax.plot(ix_r_IC_Pijnappels, omc_data["pos"][ix_r_IC_Pijnappels,2,omc_data["marker_location"].tolist().index('r_heel')], 'x', c=(1, 0, 0))
        ax.plot(ix_r_FC_Zeni, omc_data["pos"][ix_r_FC_Zeni,2,omc_data["marker_location"].tolist().index('r_toe')], 'o', mfc='none', mec=(1, 0, 0))
        ax.set_xlim((0, omc_data['pos'].shape[0]))
        ax.set_xlabel('time (in samples)')
        ax.set_ylabel('vertical position (in mm)')
        plt.show()
    
    return (ix_l_IC_Pijnappels, ix_l_FC_Zeni, ix_r_IC_Pijnappels, ix_r_FC_Zeni), (ix_start, ix_end)

def visualize_data(omc_data, imu_data, ix_l_IC, ix_l_FC, ix_r_IC, ix_r_FC, ix_start, ix_end):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(2, 1, figsize=(14.8, 10.5))

    # OMC data
    ix_l_heel = omc_data["marker_location"].tolist().index('l_heel')
    ix_r_heel = omc_data["marker_location"].tolist().index('r_heel')
    ix_l_toe = omc_data["marker_location"].tolist().index('l_toe')
    ix_r_toe = omc_data["marker_location"].tolist().index('r_toe')

    mi = np.min([np.nanmin(omc_data['pos'][:,2,ix_l_heel]), np.nanmin(omc_data['pos'][:,2,ix_l_toe]), np.nanmin(omc_data['pos'][:,2,ix_r_heel]), np.nanmin(omc_data['pos'][:,2,ix_r_toe])])
    ma = np.max([np.nanmax(omc_data['pos'][:,2,ix_l_heel]), np.nanmax(omc_data['pos'][:,2,ix_l_toe]), np.nanmax(omc_data['pos'][:,2,ix_r_heel]), np.nanmax(omc_data['pos'][:,2,ix_r_toe])])

    axs[0].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*ma, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[0].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*mi, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[0].fill_between(np.arange(ix_end, omc_data['pos'].shape[0]), np.ones((omc_data['pos'].shape[0]-ix_end,))*ma, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[0].fill_between(np.arange(ix_end, omc_data['pos'].shape[0]), np.ones((omc_data['pos'].shape[0]-ix_end,))*mi, color=(0, 0, 0), alpha=0.05, ec='none')

    axs[0].plot(omc_data['pos'][:,2,ix_l_heel], '-', c=(0, 0, 1, 0.3))
    axs[0].plot(omc_data['pos'][:,2,ix_l_toe], ':', c=(0, 0, 1, 0.3))
    axs[0].plot(omc_data['pos'][:,2,ix_r_heel], '-', c=(1, 0, 0, 0.3))
    axs[0].plot(omc_data['pos'][:,2,ix_r_toe], ':', c=(1, 0, 0, 0.3))
    axs[0].set_xlim((0, omc_data['pos'].shape[0]))
    axs[0].set_xlabel('time (in samples)')
    axs[0].set_ylabel('vertical position (in mm)')
    
    # IMU data
    ix_left_ankle = imu_data['imu_location'].tolist().index('left_ankle')
    ix_right_ankle = imu_data['imu_location'].tolist().index('right_ankle')

    mi_ = np.min([np.nanmin(imu_data['gyro'][:,2,ix_left_ankle]), np.nanmin(imu_data['gyro'][:,2,ix_right_ankle])])
    ma_ = np.max([np.nanmax(imu_data['gyro'][:,2,ix_left_ankle]), np.nanmax(imu_data['gyro'][:,2,ix_right_ankle])])

    axs[1].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*ma_, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[1].fill_between(np.arange(0, ix_start), np.ones((ix_start,))*mi_, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[1].fill_between(np.arange(ix_end, imu_data['gyro'].shape[0]), np.ones((imu_data['gyro'].shape[0]-ix_end,))*ma, color=(0, 0, 0), alpha=0.05, ec='none')
    axs[1].fill_between(np.arange(ix_end, imu_data['gyro'].shape[0]), np.ones((imu_data['gyro'].shape[0]-ix_end,))*mi, color=(0, 0, 0), alpha=0.05, ec='none')

    axs[1].plot(imu_data['gyro'][:,ix_left_ankle*3+2], '-', c=(0, 0, 1, 0.3))
    axs[1].plot(imu_data['gyro'][:,ix_right_ankle*3+2], '-', c=(1, 0, 0, 0.3))

    axs[1].set_xlim((0, imu_data['gyro'].shape[0]))
    axs[1].set_xlabel('time (in samples)')
    axs[1].set_ylabel('gyro (in degrees/s)')

    # Gait events
    for i in range(len(ix_l_IC)):
        axs[0].plot([ix_l_IC[i], ix_l_IC[i]], [mi, ma], '-', c=(0, 0, 1))
        axs[1].plot([ix_l_IC[i], ix_l_IC[i]], [mi_, ma_], '-', c=(0, 0, 1))
    for i in range(len(ix_l_FC)):
        axs[0].plot([ix_l_FC[i], ix_l_FC[i]], [mi, ma], ':', c=(0, 0, 1))
        axs[1].plot([ix_l_FC[i], ix_l_FC[i]], [mi_, ma_], ':', c=(0, 0, 1))
    for i in range(len(ix_r_IC)):
        axs[0].plot([ix_r_IC[i], ix_r_IC[i]], [mi, ma], '-', c=(1, 0, 0))
        axs[1].plot([ix_r_IC[i], ix_r_IC[i]], [mi_, ma_], '-', c=(1, 0, 0))
    for i in range(len(ix_r_FC)):
        axs[0].plot([ix_r_FC[i], ix_r_FC[i]], [mi, ma], ':', c=(1, 0, 0))
        axs[1].plot([ix_r_FC[i], ix_r_FC[i]], [mi_, ma_], ':', c=(1, 0, 0))
    plt.show()
    return

def get_data(input_data_dir, dataset, output_data_dir, **kwargs):
    """Gets a data array for the given filenames.
    
    Parameters
    ----------
    input_data_dir : str
        The relative or absolute path to the data.
    dataset : dictionary
        A dictionary with participant ids as keys, and a list of filenames as values.
    
    Returns
    -------
    .. : ..
    """
    import os
    from pymocap.utils import _load_file
    import numpy as np
    from pymocap.preprocessing import _resample_data
    import matplotlib.pyplot as plt

    visualize = kwargs.get("visualize", False)

    # For each participant id
    for participant_id in list(dataset.keys()):
        
        # For each filename
        for filename in dataset[participant_id]:

            # Load data
            omc_data = _load_file(os.path.join(input_data_dir, participant_id, "optical", filename))

            # Get the events (that serve as labels)
            (ix_l_IC, ix_l_FC, ix_r_IC, ix_r_FC), (ix_start, ix_end) = get_labels(omc_data, visualize=False)

            # Load data
            imu_data = _load_file(os.path.join(input_data_dir, participant_id, "imu", filename.replace("omc_", "imu_")))

            # Get indexes of relevant sensors
            ix_left_ankle = imu_data["imu_location"].tolist().index('left_ankle')
            ix_right_ankle = imu_data["imu_location"].tolist().index('right_ankle')

            # Reshape data
            acc = np.reshape(imu_data['acc'], (imu_data['acc'].shape[0], imu_data['acc'].shape[1]*imu_data['acc'].shape[2]), order='F')
            gyro = np.reshape(imu_data['gyro'], (imu_data['gyro'].shape[0], imu_data['gyro'].shape[1]*imu_data['gyro'].shape[2]), order='F')

            # Resample, if necessary
            if imu_data['fs'] != omc_data['fs']:
                acc = _resample_data(acc, imu_data['fs'], omc_data['fs'])
                gyro = _resample_data(gyro, imu_data['fs'], omc_data['fs'])
            
            # Get features, and labels
            features = np.hstack((acc[:,ix_left_ankle*3:ix_left_ankle*3+3], gyro[:,ix_left_ankle*3:ix_left_ankle*3+3], acc[:,ix_right_ankle*3:ix_right_ankle*3+3], gyro[:,ix_right_ankle*3:ix_right_ankle*3+3]))
            labels = np.zeros((features.shape[0], 4))
            labels[ix_l_IC,0] = 1
            labels[ix_l_FC,1] = 1
            labels[ix_r_IC,2] = 1
            labels[ix_r_FC,3] = 1

            # Only consider signal segments from start to end index
            features = features[ix_start:ix_end,:]
            labels = labels[ix_start:ix_end,:]

            # Save to folder
            np.save(os.path.join(output_data_dir, participant_id+"_"+filename.replace("omc_", "").replace(".mat","")), np.hstack((features, labels)))

            # Visualize
            if visualize == True:
                visualize_data(omc_data, imu_data, ix_l_IC, ix_l_FC, ix_r_IC, ix_r_FC, ix_start, ix_end)
    return
    
def generate_dataset():
    """Generate a dataset that is compatible with TensorFlow Keras.
    
    
    
    """
    import sys
    import os
    
    # Set input data directory
    if sys.platform == "win32":
        input_data_dir = "Z:\\Keep Control\\Data\\lab dataset"
        input_json_file_name = "Z:\\Keep Control\\Output\\lab dataset\\dataset.json"
        input_csv_file_name = "Z:\\Keep Control\\Output\\lab dataset\\participants.csv"
        output_data_dir = "Z:\\Keep Control\\Data\\lab dataset\\processed"
    elif sys.platform == "linux":
        input_data_dir = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"
        input_json_file_name = "/mnt/neurogeriatrics_data/Keep Control/Output/lab dataset/dataset.json"
        input_csv_file_name = "/mnt/neurogeriatrics_data/Keep Control/Output/lab dataset/participants.csv"
        output_data_dir = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/processed"
    else:
        raise ValueError(f"Unknown system platform: {sys.platform}")
        
    # Get folders from input data directory
    participant_ids = [folder_name for folder_name in os.listdir(input_data_dir) if folder_name.startswith("pp")]
    
    # Split files into training, validation and test set, stratified by participant type.
    train_files, val_files, test_files = split_train_val_test_files(input_data_dir, input_csv_file_name, input_json_file_name)

    get_data(input_data_dir, train_files, os.path.join(output_data_dir, 'train'), visualize=False)    
    get_data(input_data_dir, val_files, os.path.join(output_data_dir, 'val'), visualize=False)
    get_data(input_data_dir, test_files, os.path.join(output_data_dir, 'test'), visualize=False)
    return 
   
if __name__ == "__main__":
    generate_dataset()
