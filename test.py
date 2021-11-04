from pymocap.algo import predict_missing_markers
from pymocap.algo.zeni import zeni
from pymocap.utils import _load_file
from pymocap import preprocessing
from pymocap.algo import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import find_peaks

# Set data directory
if sys.platform == "win32":
    data_dir = "Z:\\Keep Control\\Data\\lab dataset"
else:
    data_dir = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset"
participant_id = "pp138"
# trial_name = "walk_fast"
# trial_name = "walk_preferred"
# trial_name = "walk_slow"
trial_names = ["walk_fast_on", "walk_preferred_on", "walk_slow_on"]


for trial_name in trial_names:
    # Set file names
    omc_file_name = "omc_" + trial_name
    imu_file_name = "imu_" + trial_name

    # Load data
    omc_data = _load_file(os.path.join(data_dir, participant_id, "optical", omc_file_name))
    imu_data = _load_file(os.path.join(data_dir, participant_id, "imu", imu_file_name))

    # ------------------------------
    # Inertial Measurement Unit Data
    # ------------------------------
    # Check if data is available for ankle IMUs
    if ("left_ankle" not in imu_data["imu_location"]) or ("right_ankle" not in imu_data["imu_location"]):
        raise Exception('Data from either the left or right IMU is missing!')
        continue

    # Reshape data
    acc = np.reshape(imu_data['acc'], (imu_data['acc'].shape[0], imu_data['acc'].shape[1]*imu_data['acc'].shape[2]), order='F')
    gyro = np.reshape(imu_data['gyro'], (imu_data['gyro'].shape[0], imu_data['gyro'].shape[1]*imu_data['gyro'].shape[2]), order='F')
    magn = np.reshape(imu_data['magn'], (imu_data['magn'].shape[0], imu_data['magn'].shape[1]*imu_data['magn'].shape[2]), order='F')

    # Indexes corresponding to left and right ankle IMU
    ix_l_ankle_IMU = np.argwhere(imu_data["imu_location"]=="left_ankle")[:,0][0]
    ix_r_ankle_IMU = np.argwhere(imu_data["imu_location"]=="right_ankle")[:,0][0]

    # Detect ICs and FCs
    ix_l_IC_IMU, ix_l_FC_IMU = salarian(-gyro[:,ix_l_ankle_IMU*3+2], imu_data['fs'])
    ix_r_IC_IMU, ix_r_FC_IMU = salarian(gyro[:,ix_r_ankle_IMU*3+2], imu_data['fs'])

    # ------------------------------
    # Optical Motion Capture Data
    # ------------------------------
    # Check if data is available for relevant markers
    my_markers = ["l_heel", "r_heel", "l_toe", "r_toe", "l_psis", "r_psis"]
    for marker in my_markers:
        if (marker not in omc_data["marker_location"]):
            print(f"Data from {marker:s} not available from {omc_file_name:s} of {participant_id:s}. Skip analysis.")

    # Indexes corresponding to left and right heel and toe markers
    markers = ["l_heel", "r_heel", "l_toe", "r_toe", "l_psis", "r_psis"]
    ix_l_heel = np.argwhere(omc_data["marker_location"]=="l_heel")[:,0][0]
    ix_r_heel = np.argwhere(omc_data["marker_location"]=="r_heel")[:,0][0]
    ix_l_toe = np.argwhere(omc_data["marker_location"]=="l_toe")[:,0][0]
    ix_r_toe = np.argwhere(omc_data["marker_location"]=="r_toe")[:,0][0]
    ix_l_psis = np.argwhere(omc_data["marker_location"]=="l_psis")[:,0][0]
    ix_r_psis = np.argwhere(omc_data["marker_location"]=="r_psis")[:,0][0]
    # ix_l_asis = np.argwhere(omc_data["marker_location"]=="l_asis")[:,0][0]
    # ix_r_asis = np.argwhere(omc_data["marker_location"]=="r_asis")[:,0][0]
    start_1 = np.nanmean(omc_data["pos"][:,:3,np.argwhere(omc_data["marker_location"]=="start_1")[:,0][0]], axis=0)
    start_2 = np.nanmean(omc_data["pos"][:,:3,np.argwhere(omc_data["marker_location"]=="start_2")[:,0][0]], axis=0)
    end_1 = np.nanmean(omc_data["pos"][:,:3,np.argwhere(omc_data["marker_location"]=="end_1")[:,0][0]], axis=0)
    end_2 = np.nanmean(omc_data["pos"][:,:3,np.argwhere(omc_data["marker_location"]=="end_2")[:,0][0]], axis=0)

    # Reshape data
    pos = np.reshape(omc_data["pos"][:,:3,:], (omc_data["pos"].shape[0], (omc_data["pos"].shape[1]-1)*omc_data["pos"].shape[2]), order='F')

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

    # ------------------------------
    # Visualize
    # ------------------------------
    # Plot signals for left and right ankle gyroscope
    fig, axs = plt.subplots(3, 1, figsize=(16, 8))
    axs[0].plot(pos[:,ix_l_heel*3+2], ls='-', c=(0, 0, 1, 0.3), label='heel (left)')
    axs[0].plot(pos[:,ix_l_toe*3+2], ls='-.', c=(0, 0, 1, 0.3), label='toe (left)')
    axs[0].plot(pos[:,ix_r_heel*3+2], ls='-', c=(1, 0, 0, 0.3), label='heel (right)')
    axs[0].plot(pos[:,ix_r_toe*3+2], ls='-.', c=(1, 0, 0, 0.3), label='toe (right)')
    axs[0].plot(ix_l_IC_OConnor, pos[ix_l_IC_OConnor,ix_l_heel*3+2], 'o', mfc='r', mec='r', ms=6)
    axs[0].plot(ix_l_FC_OConnor, pos[ix_l_FC_OConnor,ix_l_toe*3+2], 'o', mfc='none', mec='r', ms=6)
    axs[0].plot(ix_l_IC_Pijnappels, pos[ix_l_IC_Pijnappels,ix_l_heel*3+2], 'v', mfc=(0, 0.5, 0), mec=(0, 0.5, 0), ms=6)
    axs[0].plot(ix_l_FC_Pijnappels, pos[ix_l_FC_Pijnappels,ix_l_toe*3+2], 'v', mfc='none', mec=(0, 0.5, 0), ms=6)
    axs[0].set_xlim((0, pos.shape[0]))

    axs[2].plot(gyro[:,ix_l_ankle_IMU*3+2], ls='-', c=(0, 0, 1, 0.3), label='gyro (left)')
    axs[2].plot(gyro[:,ix_r_ankle_IMU*3+2], ls='-', c=(1, 0, 0, 0.3), label='gyro (right)')
    axs[2].plot(ix_l_IC_IMU, gyro[ix_l_IC_IMU,ix_l_ankle_IMU*3+2], ls='none', marker='o', mfc='b', mec='b', ms=4)
    axs[2].plot(ix_l_FC_IMU, gyro[ix_l_FC_IMU,ix_l_ankle_IMU*3+2], ls='none', marker='s', mfc='b', mec='b', ms=4)
    axs[2].plot(ix_r_IC_IMU, gyro[ix_r_IC_IMU,ix_r_ankle_IMU*3+2], ls='none', marker='o', mfc='r', mec='r', ms=4)
    axs[2].plot(ix_r_FC_IMU, gyro[ix_r_FC_IMU,ix_r_ankle_IMU*3+2], ls='none', marker='s', mfc='r', mec='r', ms=4)
    axs[2].set_xlim((0, gyro.shape[0]))

    # for i in range(len(ix_l_IC_OConnor)):
    #     axs[0].plot([ix_l_IC_OConnor[i], ix_l_IC_OConnor[i]], [np.nanmin(pos[:,ix_l_heel*3+2]),np.nanmax(pos[:,ix_l_heel*3+2])], ls='-', lw=1, c=(0, 0, 1))
    #     axs[2].plot([ix_l_IC_OConnor[i], ix_l_IC_OConnor[i]], [np.min(gyro[:,ix_l_ankle_IMU*3+2]),np.max(gyro[:,ix_l_ankle_IMU*3+2])], ls='-', lw=1, c=(0, 0, 1))
    # for i in range(len(ix_l_FC_OConnor)):
    #     axs[0].plot([ix_l_FC_OConnor[i], ix_l_FC_OConnor[i]], [np.nanmin(pos[:,ix_l_heel*3+2]),np.nanmax(pos[:,ix_l_heel*3+2])], ls=':', lw=1, c=(0, 0, 1))
    #     axs[2].plot([ix_l_FC_OConnor[i], ix_l_FC_OConnor[i]], [np.min(gyro[:,ix_l_ankle_IMU*3+2]),np.max(gyro[:,ix_l_ankle_IMU*3+2])], ls=':', lw=1, c=(0, 0, 1))

    # for i in range(len(ix_r_IC_OConnor)):
    #     axs[0].plot([ix_r_IC_OConnor[i], ix_r_IC_OConnor[i]], [np.nanmin(pos[:,ix_r_heel*3+2]),np.nanmax(pos[:,ix_r_heel*3+2])], ls='-', lw=1, c=(1, 0, 0))
    #     axs[2].plot([ix_r_IC_OConnor[i], ix_r_IC_OConnor[i]], [np.min(gyro[:,ix_r_ankle_IMU*3+2]),np.max(gyro[:,ix_r_ankle_IMU*3+2])], ls='-', lw=1, c=(1, 0, 0))
    # for i in range(len(ix_r_FC_OConnor)):
    #     axs[0].plot([ix_r_FC_OConnor[i], ix_r_FC_OConnor[i]], [np.nanmin(pos[:,ix_r_heel*3+2]),np.nanmax(pos[:,ix_r_heel*3+2])], ls=':', lw=1, c=(1, 0, 0))
    #     axs[2].plot([ix_r_FC_OConnor[i], ix_r_FC_OConnor[i]], [np.min(gyro[:,ix_r_ankle_IMU*3+2]),np.max(gyro[:,ix_r_ankle_IMU*3+2])], ls=':', lw=1, c=(1, 0, 0))

    fig.canvas.manager.set_window_title(participant_id + ": " + trial_name)
    plt.show()