import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lib.utils import _load_file
from lib.preprocessing import _predict_missing_markers, _butter_lowpass
from lib.analysis import _get_gait_events_from_OMC
import os
from scipy.signal import find_peaks

def main():
    """Run the main method.
    """

    # Set data directory
    PARENT_FOLDER = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"

    # Get a list of participant ids
    participant_ids = [folder_name for folder_name in os.listdir(PARENT_FOLDER) if folder_name.startswith("pp")]

    # Loop over the participants
    for participant_id in participant_ids[5:6]:
        print(f"{participant_id}")

        # Get a list of optical motion capture files
        omc_filenames = [file_name for file_name in os.listdir(os.path.join(PARENT_FOLDER, participant_id, "optical")) if file_name.endswith(".mat")]

        # Select only the relevant walking trial
        ix_omc_filename = [ix for ix in range(len(omc_filenames)) if ("walk_preferred" in omc_filenames[ix])]
        if len(ix_omc_filename) > 0:
            ix_omc_filename = ix_omc_filename[0]
            omc_filename = omc_filenames[ix_omc_filename]

            # Check if there equist an equivalent IMU file
            if os.path.isfile(os.path.join(PARENT_FOLDER, participant_id, "imu", omc_filename.replace("omc_", "imu_"))):
                imu_filename = omc_filename.replace("omc_", "imu_")
                
                # Load the data
                omc_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "optical", omc_filename))
                imu_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "imu", imu_filename))

                # Process optical motion capture data
                # 0. Get sampling frequency
                fs = omc_data["fs"]

                # 1. Fill gaps in the marker data
                M = omc_data["pos"][:,:3,:]
                n_time_steps, n_dimensions, n_markers = M.shape
                M = np.reshape(M, (n_time_steps, 3*n_markers), order="F")
                M = _predict_missing_markers(M)

                # 2. Low-pass filter the gap-free marker data
                M = _butter_lowpass(M, fs)
                M = np.reshape(M, (n_time_steps, n_dimensions, n_markers), order="F")  # reshape back in original .mat format

                # 3.1. Detect gait events
                l_heel_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="l_heel")[:,0][0]]
                l_toe_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="l_toe")[:,0][0]]
                l_psis_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="l_psis")[:,0][0]]
                l_asis_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="l_asis")[:,0][0]]
                r_heel_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="r_heel")[:,0][0]]
                r_toe_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="r_toe")[:,0][0]]
                r_psis_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="r_psis")[:,0][0]]
                r_asis_pos = M[:,:,np.argwhere(omc_data["marker_location"]=="r_asis")[:,0][0]]
                l_ix_IC, _, r_ix_IC, _ = _get_gait_events_from_OMC(M, fs, omc_data["marker_location"], method="OConnor")
                _, l_ix_FC, _, r_ix_FC = _get_gait_events_from_OMC(M, fs, omc_data["marker_location"], method="Zeni")

                # 3.2. Detect start and end of 5 meters within the current trial
                mid_psis_pos = ( l_psis_pos + r_psis_pos ) / 2
                start_1 = M[:,:,np.argwhere(omc_data["marker_location"]=="start_1")[:,0][0]]
                start_2 = M[:,:,np.argwhere(omc_data["marker_location"]=="start_2")[:,0][0]]
                end_1 = M[:,:,np.argwhere(omc_data["marker_location"]=="end_1")[:,0][0]]
                end_2 = M[:,:,np.argwhere(omc_data["marker_location"]=="end_2")[:,0][0]]
                start_, end_ = (start_1 + start_2) / 2, (end_1 + end_2) / 2
                dst_2_start = np.sqrt(np.sum(((mid_psis_pos-start_)**2)[:,:2], axis=1))  # distance to start
                dst_2_end = np.sqrt(np.sum(((mid_psis_pos - end_)**2)[:,:2], axis=1))    # distance to end
                ix_start_, ix_end_ = np.argmin(dst_2_start), np.argmin(dst_2_end)
                left_ix_IC = l_ix_IC[np.argwhere(np.logical_and(l_ix_IC > ix_start_, l_ix_IC < ix_end_))[:,0]]
                left_ix_FC = l_ix_FC[np.argwhere(np.logical_and(l_ix_FC > ix_start_, l_ix_FC < ix_end_))[:,0]]
                right_ix_IC = r_ix_IC[np.argwhere(np.logical_and(r_ix_IC > ix_start_, r_ix_IC < ix_end_))[:,0]]
                right_ix_FC = r_ix_FC[np.argwhere(np.logical_and(r_ix_FC > ix_start_, r_ix_FC < ix_end_))[:,0]]
                print(l_ix_IC)

                # 3.3. Visualize
                iplot = True
                if iplot == 1:
                    fig, axs = plt.subplots(3, 1, figsize=(12.8, 9.6))
                    axs[0].plot(l_heel_pos[:,2], ls='-', c=(0, 0, 1, 0.1), lw=2)
                    axs[0].plot(np.arange(ix_start_, ix_end_), l_heel_pos[ix_start_:ix_end_,2], ls='-', c=(0, 0, 1, 1), lw=1, label='left heel')
                    axs[0].plot(l_toe_pos[:,2], ls='-.', c=(0, 0, 1, 0.1), lw=2)
                    axs[0].plot(np.arange(ix_start_, ix_end_), l_toe_pos[ix_start_:ix_end_,2], ls='-.', c=(0, 0, 1, 1), lw=1, label='left toe')
                    axs[0].plot(r_heel_pos[:,2], ls='-', c=(1, 0, 0, 0.1), lw=2)
                    axs[0].plot(np.arange(ix_start_, ix_end_), r_heel_pos[ix_start_:ix_end_,2], ls='-', c=(1, 0, 0, 1), lw=1, label='right heel')
                    axs[0].plot(r_toe_pos[:,2], ls='-.', c=(1, 0, 0, 0.1), lw=2)
                    axs[0].plot(np.arange(ix_start_, ix_end_), r_toe_pos[ix_start_:ix_end_,2], ls='-.', c=(1, 0, 0, 1), lw=1, label='right toe')

                    axs[0].plot(l_ix_IC, l_heel_pos[l_ix_IC,2], ls='none', marker='o', mec=(0, 0, 1, 0.1), mfc=(0, 0, 1, 0.1))
                    axs[0].plot(left_ix_IC, l_heel_pos[left_ix_IC,2], ls='none', marker='o', mec=(0, 0, 1, 1), mfc=(0, 0, 1, 1), label='left IC')
                    # axs[0].plot(l_ix_FC, l_toe_pos[l_ix_FC,2], ls='none', marker='o', mec=(0, 0, 1, 0.1), mfc='none')
                    # axs[0].plot(left_ix_FC, l_toe_pos[left_ix_FC,2], ls='none', marker='o', mec=(0, 0, 1, 1), mfc='none', label='left FC')
                    # axs[0].plot(r_ix_IC, r_heel_pos[r_ix_IC,2], ls='none', marker='o', mec=(1, 0, 0, 0.1), mfc=(1, 0, 0, 0.1))
                    # axs[0].plot(right_ix_IC, r_heel_pos[right_ix_IC,2], ls='none', marker='o', mec=(1, 0, 0, 1), mfc=(1, 0, 0, 1), label='right IC')
                    # axs[0].plot(r_ix_FC, r_toe_pos[r_ix_FC,2], ls='none', marker='o', mec=(1, 0, 0, 0.1), mfc='none')
                    # axs[0].plot(right_ix_FC, r_toe_pos[right_ix_FC,2], ls='none', marker='o', mec=(1, 0, 0, 1), mfc='none', label='right FC')
                    
                    # Plot top view
                    axs[1].plot(np.mean(start_1[:,0]), np.mean(start_1[:,1]), ls='none', marker='^', mec='k', mfc='none')
                    axs[1].plot(np.mean(start_2[:,0]), np.mean(start_2[:,1]), ls='none', marker='^', mec='k', mfc='k')
                    axs[1].plot(np.mean(end_1[:,0]), np.mean(end_1[:,1]), ls='none', marker='v', mec='k', mfc='none')
                    axs[1].plot(np.mean(end_2[:,0]), np.mean(end_2[:,1]), ls='none', marker='v', mec='k', mfc='k')
                    axs[1].plot(mid_psis_pos[ix_start_,0], mid_psis_pos[ix_start_,1], ls='none', marker='o', mfc='none', mec='k')
                    axs[1].plot(mid_psis_pos[ix_end_,0], mid_psis_pos[ix_end_,1], ls='none', marker='o', mfc='k', mec='k')
                    axs[1].plot(mid_psis_pos[:,0], mid_psis_pos[:,1], ls='-', c=(0, 0, 0, 0.2), lw=2)
                    axs[1].plot([mid_psis_pos[ix_start_,0], mid_psis_pos[ix_end_,0]], [mid_psis_pos[ix_start_,1], mid_psis_pos[ix_end_,1]], ':', c=(0, 0, 0, 0.8))
                    axs[1].plot(mid_psis_pos[ix_start_:ix_end_,0], mid_psis_pos[ix_start_:ix_end_,1], ls='-', c=(0, 0, 0, 1), lw=1, label='mid psis')

                    # Define the direction vector
                    v = mid_psis_pos[ix_end_,:] - mid_psis_pos[ix_start_,:]  # shape: (3,)
                    print(f"Shape of v: {v.shape}")

                    # Calculate the relative position vector
                    p = r_heel_pos - mid_psis_pos  # shape: (N, 3)
                    print(f"Shape of p: {p.shape}")

                    # Project the relative position vector onto the direction vector
                    w = np.copy(p)
                    for ti in range(p.shape[0]):
                        w[ti,:2] = ( np.dot(p[ti,:2].T, v[:2]) / np.dot(v[:2].T, v[:2]) ) * v[:2]

                    axs[2].plot(np.arange(p.shape[0]), p[:,0], '-', c=(0, 0, 0, 0.5))
                    axs[2].plot(np.arange(w.shape[0]), w[:,0], '-', c=(0, 0, 0, 1))
                    # axs[1].plot(l_heel_pos[left_ix_IC,0], l_heel_pos[left_ix_IC,1], ls='none', marker='o', mec=(0, 0, 1, 1), mfc=(0, 0, 1, 1))
                    # axs[1].plot(r_heel_pos[right_ix_IC,0], r_heel_pos[right_ix_IC,1], ls='none', marker='o', mec=(1, 0, 0, 1), mfc=(1, 0, 0, 1))
                    
                    # Labels
                    axs[0].set_xlabel('time (samples)')
                    axs[0].set_ylabel('Z position (mm)')
                    axs[1].set_xlabel('X position (mm)')
                    axs[1].set_ylabel('Y position (mm)')
                    axs[2].set_xlabel('time (samples)')
                    axs[2].set_ylabel('rel X position (mm)')

                    axs[0].legend()
                    axs[1].legend()

                    # 
                    # plt.savefig(os.path.join("/home/robr/Desktop/figures", participant_id + ".png"), dpi=300, transparent=True)
                    plt.show()

    return
                

if __name__ == "__main__":
    main()