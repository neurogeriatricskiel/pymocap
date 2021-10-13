import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lib.utils import _load_file
from lib.preprocessing import _predict_missing_markers, _butter_lowpass, _align_trajectories_with_walking_direction, _get_start_end_index
from lib.analysis import _get_gait_events_from_OMC, _get_gait_evens_from_IMU, _extract_temporal_gait_params
import os
from scipy.signal import find_peaks

def main():
    """Run main.
    """

    # Set data directory
    # PARENT_FOLDER = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"  # on Linux
    PARENT_FOLDER = "Z:\\Keep Control\\Data\\lab dataset"

    # Get a list of participant ids
    participant_ids = [folder_name for folder_name in os.listdir(PARENT_FOLDER) if folder_name.startswith("pp")]

    # Trial
    trial_name = "walk_preferred"

    # Loop over the participants
    for (ix_participant, participant_id) in enumerate(participant_ids[8:10]):
        print(f"{participant_id}")

        # Get a list of optical motion capture files
        omc_filenames = [file_name for file_name in os.listdir(os.path.join(PARENT_FOLDER, participant_id, "optical")) if file_name.endswith(".mat")]

        # Select only the relevant walking trial
        ix_omc_filename = [ix for ix in range(len(omc_filenames)) if (trial_name in omc_filenames[ix])]
        if len(ix_omc_filename) > 0:
            ix_omc_filename = ix_omc_filename[0]
            omc_filename = omc_filenames[ix_omc_filename]

            # Check if there equist an equivalent IMU file
            if os.path.isfile(os.path.join(PARENT_FOLDER, participant_id, "imu", omc_filename.replace("omc_", "imu_"))):
                imu_filename = omc_filename.replace("omc_", "imu_")
                
                # Load the data
                omc_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "optical", omc_filename))
                imu_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "imu", imu_filename))

                # Preprocess optical motion capture data
                # 0. Get sampling frequency, raw marker data, and data dimensions
                fs_omc = omc_data["fs"]
                raw_data = omc_data["pos"][:,:3,:]
                n_time_steps, n_dimensions, n_markers = raw_data.shape
                
                # 1. Fill gaps in marker trajectories
                raw_data = np.reshape(raw_data, (n_time_steps, n_dimensions * n_markers), order="F")
                filled_data = _predict_missing_markers(raw_data)

                # 2. Low-pass filter the gap-free marker data
                filtered_data = _butter_lowpass(filled_data, fs_omc)
                filtered_data = np.reshape(filtered_data, (n_time_steps, n_dimensions, n_markers), order="F")
                ix_start, ix_end = _get_start_end_index(filtered_data, omc_data["marker_location"])

                # 3. Align data with main direction of walking
                aligned_data = _align_trajectories_with_walking_direction(filtered_data, omc_data["marker_location"])

                # 4. Detect gait events
                l_ix_IC, _, r_ix_IC, _ = _get_gait_events_from_OMC(aligned_data, fs_omc, omc_data["marker_location"], method="Pijnappels")
                _, l_ix_FC, _, r_ix_FC = _get_gait_events_from_OMC(aligned_data, fs_omc, omc_data["marker_location"], method="ZeniJr")
                left_ix_IC = l_ix_IC[np.logical_and(l_ix_IC >= ix_start, l_ix_IC <= ix_end)]
                left_ix_FC = l_ix_FC[np.logical_and(l_ix_FC >= ix_start, l_ix_FC <= ix_end)]
                right_ix_IC = r_ix_IC[np.logical_and(r_ix_IC >= ix_start, r_ix_IC <= ix_end)]
                right_ix_FC = r_ix_FC[np.logical_and(r_ix_FC >= ix_start, r_ix_FC <= ix_end)]

                # 5. Visualize
                iplot = True
                if iplot == True:

                    # Get relevant position markers
                    l_psis_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="l_psis")[:,0]], axis=-1)
                    r_psis_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="r_psis")[:,0]], axis=-1)
                    l_asis_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="l_asis")[:,0]], axis=-1)
                    r_asis_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="r_asis")[:,0]], axis=-1)
                    pelvis_pos = ( l_psis_pos + r_psis_pos + l_asis_pos + r_asis_pos ) / 4
                    l_heel_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="l_heel")[:,0]], axis=-1)
                    r_heel_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="r_heel")[:,0]], axis=-1)
                    l_toe_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="l_toe")[:,0]], axis=-1)
                    r_toe_pos = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="r_toe")[:,0]], axis=-1)
                    start_1 = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="start_1")[:,0]], axis=-1)
                    start_2 = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="start_2")[:,0]], axis=-1)
                    end_1 = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="end_1")[:,0]], axis=-1)
                    end_2 = np.squeeze(aligned_data[:,:,np.argwhere(omc_data["marker_location"]=="end_2")[:,0]], axis=-1)

                    fig, axs = plt.subplots(2, 1, figsize=(19.2, 9.6))
                    # Top view
                    # -- Start and end cones
                    axs[0].plot(np.mean(start_1[:,0]), np.mean(start_1[:,1]), 'o', mfc='none', mec=(0, 0, 0, 1), ms=6)
                    axs[0].plot(np.mean(start_2[:,0]), np.mean(start_2[:,1]), 'o', mfc=(0, 0, 0, 1), mec=(0, 0, 0, 1), ms=6)
                    axs[0].plot(np.mean(end_1[:,0]), np.mean(end_1[:,1]), 's', mfc='none', mec=(0, 0, 0, 1), ms=6)
                    axs[0].plot(np.mean(end_2[:,0]), np.mean(end_2[:,1]), 's', mfc=(0, 0, 0, 1), mec=(0, 0, 0, 1), ms=6)
                    # -- Virtual pelvis marker
                    axs[0].plot(pelvis_pos[:,0], pelvis_pos[:,1], '-', c=(0, 0, 0, 0.2), lw=4)
                    axs[0].plot(pelvis_pos[ix_start:ix_end,0], pelvis_pos[ix_start:ix_end,1], '-', c=(0, 0, 0, 1), lw=1)
                    # -- Left heel/toe marker
                    axs[0].plot(l_heel_pos[:,0], l_heel_pos[:,1], '-', c=(0, 0, 1, 0.2), lw=4)
                    axs[0].plot(l_heel_pos[l_ix_IC,0], l_heel_pos[l_ix_IC,1], 'v', mfc='none', mec=(0, 0, 1, 1), ms=10)
                    axs[0].plot(l_toe_pos[:,0], l_toe_pos[:,1], '-.', c=(0, 0, 1, 0.1), lw=2)
                    axs[0].plot(l_toe_pos[l_ix_FC,0], l_toe_pos[l_ix_FC,1], '^', mfc='none', mec=(0, 0, 1, 1), ms=8)
                    # -- Right heel marker
                    axs[0].plot(r_heel_pos[:,0], r_heel_pos[:,1], '-', c=(1, 0, 0, 0.2), lw=4)
                    axs[0].plot(r_heel_pos[r_ix_IC,0], r_heel_pos[r_ix_IC,1], 'v', mfc='none', mec=(1, 0, 0, 1), ms=10)
                    axs[0].plot(r_toe_pos[:,0], r_toe_pos[:,1], '-.', c=(1, 0, 0, 0.1), lw=2)
                    axs[0].plot(r_toe_pos[r_ix_FC,0], r_toe_pos[r_ix_FC,1], '^', mfc='none', mec=(1, 0, 0, 1), ms=8)
                    axs[0].set_xlabel('X position (mm)')
                    axs[0].set_ylabel('Y position (mm)')
                    axs[0].spines['right'].set_visible(False)
                    axs[0].spines['top'].set_visible(False)

                    # Sagittal view
                    # -- Left heel/toe marker
                    axs[1].plot(np.arange(aligned_data.shape[0]), l_heel_pos[:,2], '-', c=(0, 0, 1, 0.2), lw=4)
                    axs[1].plot(np.arange(ix_start, ix_end), l_heel_pos[ix_start:ix_end,2], '-', c=(0, 0, 1, 1), lw=1)
                    axs[1].plot(l_ix_IC, l_heel_pos[l_ix_IC,2], 'v', mfc='none', mec=(0, 0, 1, 0.6), ms=8)
                    axs[1].plot(left_ix_IC, l_heel_pos[left_ix_IC,2], 'v', mfc=(0, 0, 1, 1), mec=(0, 0, 1, 1), ms=8)
                    axs[1].plot(np.arange(aligned_data.shape[0]), l_toe_pos[:,2], '-.', c=(0, 0, 1, 0.2), lw=4)
                    axs[1].plot(np.arange(ix_start, ix_end), l_toe_pos[ix_start:ix_end,2], '-.', c=(0, 0, 1, 1), lw=1)
                    axs[1].plot(l_ix_FC, l_toe_pos[l_ix_FC,2], '^', mfc='none', mec=(0, 0, 1, 0.6), ms=8)
                    axs[1].plot(left_ix_FC, l_toe_pos[left_ix_FC,2], '^', mfc=(0, 0, 1, 1), mec=(0, 0, 1, 1), ms=8)
                    # -- Right heel/toe marker
                    axs[1].plot(np.arange(aligned_data.shape[0]), r_heel_pos[:,2], '-', c=(1, 0, 0, 0.2), lw=4)
                    axs[1].plot(np.arange(ix_start, ix_end), r_heel_pos[ix_start:ix_end,2], '-', c=(1, 0, 0, 1), lw=1)
                    axs[1].plot(r_ix_IC, r_heel_pos[r_ix_IC,2], 'v', mfc='none', mec=(1, 0, 0, 1), ms=8)
                    axs[1].plot(right_ix_IC, r_heel_pos[right_ix_IC,2], 'v', mfc=(1, 0, 0, 1), mec=(1, 0, 0, 1), ms=8)
                    axs[1].plot(np.arange(aligned_data.shape[0]), r_toe_pos[:,2], '-.', c=(1, 0, 0, 0.2), lw=4)
                    axs[1].plot(np.arange(ix_start, ix_end), r_toe_pos[ix_start:ix_end,2], '-.', c=(1, 0, 0, 1), lw=1)
                    axs[1].plot(r_ix_FC, r_toe_pos[r_ix_FC,2], '^', mfc='none', mec=(1, 0, 0, 1), ms=8)
                    axs[1].plot(right_ix_FC, r_toe_pos[right_ix_FC,2], '^', mfc=(1, 0, 0, 1), mec=(1, 0, 0, 1), ms=8)
                    axs[1].set_xlim(0, aligned_data.shape[0])
                    axs[1].set_xlabel('time (s)')
                    axs[1].set_ylabel('Z position (mm)')
                    axs[1].spines['right'].set_visible(False)
                    axs[1].spines['top'].set_visible(False)
                    plt.savefig(os.path.join("Z:\\Keep Control\\Output\\lab dataset", "test_figure_"+participant_id+"_"+trial_name+".png"))
                    # plt.show()
    return
                

if __name__ == "__main__":
    main()