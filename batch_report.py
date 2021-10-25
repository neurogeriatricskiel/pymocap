import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [21.0*(1/2.54), 14.8*(1/2.54)]
mpl.rcParams['figure.dpi'] = 180
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.bottom'] = False
import matplotlib.pyplot as plt
from lib.utils import _load_file
from lib.preprocessing import _predict_missing_markers, _butter_lowpass, _align_trajectories_with_walking_direction, _get_start_end_index, _get_data_from_marker
from lib.analysis import _get_gait_events_from_OMC, _get_gait_events_from_IMU, _extract_temporal_gait_params
import os
from scipy.signal import find_peaks
from lib.postprocessing import _compare_events

def main():
    """Run main.
    """

    # Set data directory
    PARENT_FOLDER = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"  # on Linux
    OUTPUT_FOLDER = "/mnt/neurogeriatrics_data/Keep Control/Output/lab dataset"  # on Linux
    # PARENT_FOLDER = "Z:\\Keep Control\\Data\\lab dataset"
    # OUTPUT_FOLDER = "Z:\\Keep Control\\Output\\lab dataset"
    
    # Get a list of participant ids
    participant_ids = [folder_name for folder_name in os.listdir(PARENT_FOLDER) if folder_name.startswith("pp")]

    # Trial
    trial_name = "walk_preferred"

    # Loop over the participants
    l_IC_results, l_FC_results, r_IC_results, r_FC_results = [], [], [], []
    for (ix_participant, participant_id) in enumerate(participant_ids[-2:]):
        print(f"{ix_participant:>3d}: {participant_id}")

        # Get a list of optical motion capture files
        omc_filenames = [file_name for file_name in os.listdir(os.path.join(PARENT_FOLDER, participant_id, "optical")) if file_name.endswith(".mat")]

        # Select only the relevant walking trial
        ix_omc_filename = [ix for ix in range(len(omc_filenames)) if (trial_name in omc_filenames[ix])]
        if len(ix_omc_filename) > 0:
            omc_filename = omc_filenames[ix_omc_filename[0]]

            # Check if there exists an equivalent IMU file
            if os.path.isfile(os.path.join(PARENT_FOLDER, participant_id, "imu", omc_filename.replace("omc_", "imu_"))):
                imu_filename = omc_filename.replace("omc_", "imu_")
                
                # Load the optical motion capture data
                omc_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "optical", omc_filename))
                fs_omc = omc_data["fs"]

                # 0. Get sampling frequency, raw marker data, and data dimensions
                n_time_steps, n_dimensions, n_markers = omc_data["pos"][:,:3,:].shape

                # 0. Check if marker data is available for all relevant markers
                l_heel_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='l_heel')
                r_heel_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='r_heel')
                l_toe_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='l_toe')
                r_toe_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='r_toe')
                l_psis_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='l_psis')
                r_psis_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='r_psis')
                l_asis_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='l_asis')
                r_asis_pos = _get_data_from_marker(omc_data["pos"][:,:3,:], omc_data["marker_location"], marker='r_asis')
                if (l_heel_pos is None) or (l_toe_pos is None) or (l_psis_pos is None) or (l_asis_pos is None) or (r_heel_pos is None) or (r_toe_pos is None) or (r_psis_pos is None) or (r_asis_pos is None):
                    continue
                
                # 1. Fill gaps in marker trajectories
                raw_data = np.reshape(omc_data["pos"][:,:3,:], (n_time_steps, n_dimensions * n_markers), order="F")
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

                # Load inertial measurement unit data
                imu_data = _load_file(os.path.join(PARENT_FOLDER, participant_id, "imu", imu_filename))
                fs_imu = imu_data["fs"]
                if fs_imu != fs_omc:
                    print(f"Resampling IMU data to match that of OMC.")
                
                # 1. Detect gait event from ankle-worn IMU
                l_ank_acc, l_ank_gyro, l_ix_MS_IMU, l_ix_IC_IMU, l_ix_FC_IMU = _get_gait_events_from_IMU(imu_data, label="left_ankle")
                r_ank_acc, r_ank_gyro, r_ix_MS_IMU, r_ix_IC_IMU, r_ix_FC_IMU = _get_gait_events_from_IMU(imu_data, label="right_ankle")
                left_ix_IC_IMU = l_ix_IC_IMU[np.logical_and(l_ix_IC_IMU >= ix_start, l_ix_IC_IMU <= ix_end)]
                left_ix_FC_IMU = l_ix_FC_IMU[np.logical_and(l_ix_FC_IMU >= ix_start, l_ix_FC_IMU <= ix_end)]
                right_ix_IC_IMU = r_ix_IC_IMU[np.logical_and(r_ix_IC_IMU >= ix_start, r_ix_IC_IMU <= ix_end)]
                right_ix_FC_IMU = r_ix_FC_IMU[np.logical_and(r_ix_FC_IMU >= ix_start, r_ix_FC_IMU <= ix_end)]

                # 2. Compare IMU events (predicted) to OMC events (annotated)
                # ---- Left initial contacts
                pointer_annotated2predicted, pointer_predicted2annotated, time_diff = _compare_events(l_ix_IC, l_ix_IC_IMU)
                j = 0
                for i in range(len(pointer_annotated2predicted)):
                    if pointer_annotated2predicted[i] > -999:    
                        while j < len(pointer_predicted2annotated):
                            if pointer_predicted2annotated[j] == i:
                                if ((l_ix_IC[i] >= ix_start) and (l_ix_IC[i] <= ix_end)) or ((l_ix_IC_IMU[j] >= ix_start) and (l_ix_IC_IMU[j] <= ix_end)):
                                    l_IC_results.append([participant_id, 1, l_ix_IC[i], l_ix_IC_IMU[j]])
                                j += 1
                                break
                            else:
                                if ((l_ix_IC_IMU[j] >= ix_start) and (l_ix_IC_IMU[j] <= ix_end)):
                                    l_IC_results.append([participant_id, 1, 'N/A', l_ix_IC_IMU[j]])
                                j += 1
                    else:
                        if ((l_ix_IC[i] >= ix_start) and (l_ix_IC[i] <= ix_end)):
                            l_IC_results.append([participant_id, 1, l_ix_IC[i], 'N/A'])

                # ---- Left final contacts
                pointer_annotated2predicted, pointer_predicted2annotated, time_diff = _compare_events(l_ix_FC, l_ix_FC_IMU)
                j = 0
                for i in range(len(pointer_annotated2predicted)):
                    if pointer_annotated2predicted[i] > -999: 
                        while j < len(pointer_predicted2annotated):
                            if pointer_predicted2annotated[j] == i:
                                if ((l_ix_FC[i] >= ix_start) and (l_ix_FC[i] <= ix_end)) or ((l_ix_FC_IMU[j] >= ix_start) and (l_ix_FC_IMU[j] <= ix_end)):
                                    l_FC_results.append([participant_id, 2, l_ix_FC[i], l_ix_FC_IMU[j]])
                                j += 1
                                break
                            else:
                                if ((l_ix_FC_IMU[j] >= ix_start) and (l_ix_FC_IMU[j] <= ix_end)):
                                    l_FC_results.append([participant_id, 2, 'N/A', l_ix_FC_IMU[j]])
                                j += 1
                    else:
                        if ((l_ix_FC[i] >= ix_start) and (l_ix_FC[i] <= ix_end)):
                            l_FC_results.append([participant_id, 2, l_ix_FC[i], 'N/A'])

                # ---- Right initial contacts
                pointer_annotated2predicted, pointer_predicted2annotated, time_diff = _compare_events(r_ix_IC, r_ix_IC_IMU)
                j = 0
                for i in range(len(pointer_annotated2predicted)):
                    if pointer_annotated2predicted[i] > -999:    
                        while j < len(pointer_predicted2annotated):
                            if pointer_predicted2annotated[j] == i:
                                if ((r_ix_IC[i] >= ix_start) and (r_ix_IC[i] <= ix_end)) or ((r_ix_IC_IMU[j] >= ix_start) and (r_ix_IC_IMU[j] <= ix_end)):
                                    r_IC_results.append([participant_id, 3, r_ix_IC[i], r_ix_IC_IMU[j]])
                                j += 1
                                break
                            else:
                                if ((r_ix_IC_IMU[j] >= ix_start) and (r_ix_IC_IMU[j] <= ix_end)):
                                    r_IC_results.append([participant_id, 3, 'N/A', r_ix_IC_IMU[j]])
                                j += 1
                    else:
                        if ((r_ix_IC[i] >= ix_start) and (r_ix_IC[i] <= ix_end)):
                            r_IC_results.append([participant_id, 3, r_ix_IC[i], 'N/A'])
                
                # ---- Right final contacts
                pointer_annotated2predicted, pointer_predicted2annotated, time_diff = _compare_events(r_ix_FC, r_ix_FC_IMU)
                j = 0
                for i in range(len(pointer_annotated2predicted)):
                    if pointer_annotated2predicted[i] > -999:    
                        while j < len(pointer_predicted2annotated):
                            if pointer_predicted2annotated[j] == i:
                                if ((r_ix_FC[i] >= ix_start) and (r_ix_FC[i] <= ix_end)) or ((r_ix_FC_IMU[j] >= ix_start) and (r_ix_FC_IMU[j] <= ix_end)):
                                    r_FC_results.append([participant_id, 4, r_ix_FC[i], r_ix_FC_IMU[j]])
                                j += 1
                                break
                            else:
                                if ((r_ix_FC_IMU[j] >= ix_start) and (r_ix_FC_IMU[j] <= ix_end)):
                                    r_FC_results.append([participant_id, 4, 'N/A', r_ix_FC_IMU[j]])
                                j += 1
                    else:
                        if ((r_ix_FC[i] >= ix_start) and (r_ix_FC[i] <= ix_end)):
                            r_FC_results.append([participant_id, 4, r_ix_FC[i], 'N/A'])
                
                # Visualize
                iplot = True
                if iplot == True:
                    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

                    axs[0].fill_between(np.arange(0, ix_start), np.ones((l_ank_gyro[:ix_start,1].shape[0],))*max(l_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[0].fill_between(np.arange(0, ix_start), np.ones((l_ank_gyro[:ix_start,1].shape[0],))*min(l_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[0].fill_between(np.arange(ix_end, l_ank_gyro.shape[0]), np.ones((l_ank_gyro[ix_end:,1].shape[0],))*max(l_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[0].fill_between(np.arange(ix_end, l_ank_gyro.shape[0]), np.ones((l_ank_gyro[ix_end:,1].shape[0],))*min(l_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')

                    axs[1].fill_between(np.arange(0, ix_start), np.ones((l_ank_gyro[:ix_start,1].shape[0],))*max(r_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[1].fill_between(np.arange(0, ix_start), np.ones((l_ank_gyro[:ix_start,1].shape[0],))*min(r_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[1].fill_between(np.arange(ix_end, l_ank_gyro.shape[0]), np.ones((l_ank_gyro[ix_end:,1].shape[0],))*max(r_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')
                    axs[1].fill_between(np.arange(ix_end, l_ank_gyro.shape[0]), np.ones((l_ank_gyro[ix_end:,1].shape[0],))*min(r_ank_gyro[:,1]), color=(0, 0, 0), alpha=0.1, ec='none')

                    # Left side (left = bLue)
                    axs[0].plot(np.arange(l_ank_gyro.shape[0]), l_ank_gyro[:,1], '-', c=(0, 0, 0, 0.3), lw=1)
                    # axs[0].plot(np.arange(ix_start,ix_end), l_ank_gyro[ix_start:ix_end,1], '-', c=(0, 0, 0), lw=1)
                    axs[0].plot(l_ix_IC, l_ank_gyro[l_ix_IC,1], 'v', mec=(0, 0, 1, 0.2), mfc='none', ms=8)
                    axs[0].plot(left_ix_IC, l_ank_gyro[left_ix_IC,1], 'v', mec=(0, 0, 1), mfc='none', ms=8)
                    for i in range(len(l_ix_IC_IMU)):
                        axs[0].plot([l_ix_IC_IMU[i], l_ix_IC_IMU[i]], [np.min(l_ank_gyro[:,1]), np.max(l_ank_gyro[:,1])], ls='--', c=(0, 0, 1, 0.2))
                    for i in range(len(left_ix_IC_IMU)):
                        axs[0].plot([left_ix_IC_IMU[i], left_ix_IC_IMU[i]], [np.min(l_ank_gyro[:,1]), np.max(l_ank_gyro[:,1])], ls='--', c=(0, 0, 1))
                    axs[0].plot(l_ix_FC, l_ank_gyro[l_ix_FC,1], 'o', mec=(0, 0, 1, 0.2), mfc='none', ms=8)
                    axs[0].plot(left_ix_FC, l_ank_gyro[left_ix_FC,1], 'o', mec=(0, 0, 1), mfc='none', ms=8)
                    for i in range(len(l_ix_FC_IMU)):
                        axs[0].plot([l_ix_FC_IMU[i], l_ix_FC_IMU[i]], [np.min(l_ank_gyro[:,1]), np.max(l_ank_gyro[:,1])], ls=':', c=(0, 0, 1, 0.2))
                    for i in range(len(left_ix_FC_IMU)):
                        axs[0].plot([left_ix_FC_IMU[i], left_ix_FC_IMU[i]], [np.min(l_ank_gyro[:,1]), np.max(l_ank_gyro[:,1])], ls=':', c=(0, 0, 1))
                    
                    # Right side (right = Red)
                    axs[1].plot(np.arange(r_ank_gyro.shape[0]), r_ank_gyro[:,1], '-', c=(0, 0, 0, 0.3), lw=1)
                    # axs[1].plot(np.arange(ix_start,ix_end), r_ank_gyro[ix_start:ix_end,1], '-', c=(1, 0, 0), lw=1)
                    axs[1].plot(r_ix_IC, r_ank_gyro[r_ix_IC,1], 'v', mec=(1, 0, 0, 0.2), mfc='none', ms=8)
                    axs[1].plot(right_ix_IC, r_ank_gyro[right_ix_IC,1], 'v', mec=(1, 0, 0), mfc='none', ms=8)
                    for i in range(len(r_ix_IC_IMU)):
                        axs[1].plot([r_ix_IC_IMU[i], r_ix_IC_IMU[i]], [np.min(r_ank_gyro[:,1]), np.max(r_ank_gyro[:,1])], '--', c=(1, 0, 0, 0.2))
                    for i in range(len(right_ix_IC_IMU)):
                        axs[1].plot([right_ix_IC_IMU[i], right_ix_IC_IMU[i]], [np.min(r_ank_gyro[:,1]), np.max(r_ank_gyro[:,1])], '--', c=(1, 0, 0))
                    axs[1].plot(r_ix_FC, r_ank_gyro[r_ix_FC,1], 'o', mec=(1, 0, 0, 0.2), mfc='none', ms=8)
                    axs[1].plot(right_ix_FC, r_ank_gyro[right_ix_FC,1], 'o', mec=(1, 0, 0), mfc='none', ms=8)
                    for i in range(len(r_ix_FC_IMU)):
                        axs[1].plot([r_ix_FC_IMU[i], r_ix_FC_IMU[i]], [np.min(r_ank_gyro[:,1]), np.max(r_ank_gyro[:,1])], ':', c=(1, 0, 0, 0.2))
                    for i in range(len(right_ix_FC_IMU)):
                        axs[1].plot([right_ix_FC_IMU[i], right_ix_FC_IMU[i]], [np.min(r_ank_gyro[:,1]), np.max(r_ank_gyro[:,1])], ':', c=(1, 0, 0))
                    
                    axs[0].set_title("Left side")
                    axs[1].set_title("Right side")
                    axs[0].set_ylabel("angular velocity (in degrees/s)")
                    axs[1].set_ylabel("angular velocity (in degrees/s)")
                    axs[1].set_xlabel("time (in samples)")
                    # plt.show()
                    plt.savefig(os.path.join(OUTPUT_FOLDER, "test_figure_"+participant_id+"_"+trial_name+".png"))

            else:
                continue
        else:
            print(f"For participant {participant_id:s} no data is available for the {trial_name:s} trial.")
    return
                

if __name__ == "__main__":
    main()