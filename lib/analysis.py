from math import dist
import warnings
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance
from lib.preprocessing import _remove_drift_200Hz, _butter_lowpass, _get_data_from_marker

def _extract_temporal_gait_params(left_ix_IC, left_ix_FC, right_ix_IC, right_ix_FC, fs):
    """Extracts temporal gait parameters based on the detected left and right initial and final contacts.

    Parameters
    ----------
    left_ix_IC : (N,) array_like
        Array of indexes corresponding to left initial contacts.
    left_ix_FC : (M,) array_like
        Likewise, but for the left final contacts.
    right_ix_IC : (K,) array_like
        Likewise, but for the right initial contacts.
    right_ix_FC : (L,) array_like
        Likewise, but for the right final contacts.
    fs : int, float
        Sampling frequency (Hz).
    
    Returns
    -------
    """
    left_stride_time, left_stance_time, left_swing_time = [], [], []

    # Loop over the events
    for i in range(len(left_ix_IC)-1):
        left_stride_time.append((left_ix_IC[i+1] - left_ix_IC[i]) / fs)  # stride time
        f = np.argwhere(np.logical_and(left_ix_FC > left_ix_IC[i], left_ix_FC < left_ix_IC[i+1]))[:,0]
        if len(f) > 0:
            left_stance_time.append((left_ix_FC[f[0]] - left_ix_IC[i]) / fs)
            left_swing_time.append((left_ix_IC[i+1] - left_ix_FC[f[0]])/fs)

    right_stride_time, right_stance_time, right_swing_time = [], [], []

    # Loop over the events
    for i in range(len(right_ix_IC)-1):
        right_stride_time.append((right_ix_IC[i+1] - right_ix_IC[i]) / fs)  # stride time
        f = np.argwhere(np.logical_and(right_ix_FC > right_ix_IC[i], right_ix_FC < right_ix_IC[i+1]))[:,0]
        if len(f) > 0:
            right_stance_time.append((right_ix_FC[f[0]] - right_ix_IC[i]) / fs)
            right_swing_time.append((right_ix_IC[i+1] - right_ix_FC[f[0]])/fs)

    double_limb_support_time, single_limb_support_time = [], []
    if left_ix_IC[0] < right_ix_IC[0]:
        print("Left before right")
        for i in range(len(left_ix_IC)-1):
            # Find the right FC following the current left IC
            f = np.argwhere(np.logical_and(right_ix_FC > left_ix_IC[i], right_ix_FC < left_ix_IC[i+1]))[:,0]
            if len(f) > 0:
                g = np.argwhere(np.logical_and(right_ix_IC > right_ix_FC[f[0]], right_ix_IC < left_ix_IC[i+1]))[:,0]
                if len(g) > 0:
                    h = np.argwhere(np.logical_and(left_ix_FC > right_ix_IC[g[0]], left_ix_FC < left_ix_IC[i+1]))[:,0]
                    if len(h) > 0:
                        double_limb_support_time.append(( (right_ix_FC[f[0]] - left_ix_IC[i]) + (left_ix_FC[h[0]] - right_ix_IC[g[0]]) ) / fs)
                        single_limb_support_time.append(( (right_ix_IC[g[0]] - right_ix_FC[f[0]]) + (left_ix_IC[i+1] - left_ix_FC[h[0]]) ) / fs)
    else:
        for i in range(len(right_ix_IC)-1):
            f = np.argwhere(np.logical_and(left_ix_FC > right_ix_IC[i], left_ix_FC < right_ix_IC[i+1]))[:,0]
            if len(f) > 0:
                g = np.argwhere(np.logical_and(left_ix_IC > left_ix_FC[f[0]], left_ix_IC < right_ix_IC[i+1]))[:,0]
                if len(g) > 0:
                    h = np.argwhere(np.logical_and(right_ix_FC > left_ix_IC[g[0]], right_ix_FC < right_ix_IC[i+1]))[:,0]
                    if len(h) > 0:
                        double_limb_support_time.append(( (left_ix_FC[f[0]] - right_ix_IC[i]) + (right_ix_FC[h[0]] - left_ix_IC[g[0]]) ) / fs)
                        single_limb_support_time.append(( (left_ix_IC[g[0]] - left_ix_FC[f[0]]) + (right_ix_IC[i+1] - right_ix_FC[h[0]]) ) / fs)
    
    # Output dictionary
    out = {"left_stride_time": np.array(left_stride_time), "left_stance_time": np.array(left_stance_time), "left_swing_time": np.array(left_swing_time), \
        "right_stride_time": np.array(right_stride_time), "right_stance_time": np.array(right_stance_time), "right_swing_time": np.array(right_swing_time), \
            "double_limb_support_time": np.array(double_limb_support_time), "single_limb_support_time": np.array(single_limb_support_time)}
    return out

def _get_gait_events_Pijnappels(heel_pos, toe_pos, fs):
    """Detect gait events from optical motion capture data according to Pijnappels et al. (2001).

    Parameters
    ----------
    heel_pos : (N, 3) array_like
        The heel marker position data.
    toe_pos : (N, 3) array_like
        The toe marker position data.
    fs : int, float
        Sampling frequency (in Hz).
    """

    # Calculate the mid foot
    mid_foot_pos = ( heel_pos + toe_pos ) / 2

    # Calculate the velocity signals
    heel_vel = np.zeros_like(heel_pos)
    heel_vel[1:,:] = ( heel_pos[1:,:] - heel_pos[:-1,:] ) / (1/fs)
    heel_vel[0,:] = heel_vel[1,:]
    
    toe_vel = np.zeros_like(toe_pos)
    toe_vel[1:,:] = ( toe_pos[1:,:] - toe_pos[:-1,:] ) / (1./fs)
    toe_vel[0,:] = toe_vel[1,:]

    mid_foot_vel = np.zeros_like(mid_foot_pos)
    mid_foot_vel[1:,:] = ( mid_foot_pos[1:,:] - mid_foot_pos[:-1,:] ) / (1./fs)
    mid_foot_vel[0,:] = mid_foot_vel[1,:]

    # Detect peaks in the velocity signals
    #   Define thresholds
    #       minimum horizontal velocity: 10% of the range in horizontal velocity
    #       minimum vertical velocity: 10% of the range in vertical veloctity
    #       minimum time between two successive peaks: 100 ms
    thr_min_vel_x = 0.1*(np.max(mid_foot_vel[:,0]) - np.min(mid_foot_vel[:,0]))
    thr_min_mid_foot_vel_z = 0.1*(np.max(mid_foot_vel[:,-1]) - np.min(mid_foot_vel[:,-1]))
    thr_min_heel_vel_z = 0.1*(np.max(heel_vel[:,-1]) - np.min(heel_vel[:,-1]))
    thr_min_toe_vel_z = 0.1*(np.max(toe_vel[:,-1]) - np.min(toe_vel[:,-1]))

    # Find (positive) peaks in the horizontal velocity
    ix_max_vel_x, _ = find_peaks(mid_foot_vel[:,0], height=thr_min_vel_x, distance=fs//4)

    # Find positive and negative peaks in the vertical velocity
    ix_max_mid_foot_vel_z, _ = find_peaks(mid_foot_vel[:,-1], height=thr_min_mid_foot_vel_z, distance=fs//10)
    ix_min_mid_foot_vel_z, _ = find_peaks(-mid_foot_vel[:,-1], height=thr_min_mid_foot_vel_z, distance=fs//10)
    ix_max_heel_vel_z, _ = find_peaks(heel_vel[:,-1], height=thr_min_heel_vel_z, distance=fs//10)
    ix_min_heel_vel_z, _ = find_peaks(-heel_vel[:,-1], height=thr_min_heel_vel_z, distance=fs//10)
    ix_max_toe_vel_z, _ = find_peaks(toe_vel[:,-1], height=thr_min_toe_vel_z, distance=fs//10)
    ix_min_toe_vel_z, _ = find_peaks(-toe_vel[:,-1], height=thr_min_toe_vel_z, distance=fs//10)

    # [...] The timing of HS (IC) correlated closely to the time of a local minimum in the vertical
    # velocity component of the toe marker. [...] (Pijnappels et al., 2001)
    # For each peak in the horizontal velocity (assumed to correspond to midswing)
    ix_IC, ix_FC = [], []
    for ix_pk in ix_max_vel_x:

        # Consider the negative peaks following the current (horizontal) peak
        f = np.argwhere(np.logical_and(ix_min_toe_vel_z > ix_pk, ix_min_toe_vel_z < ix_pk+fs//2))[:,0]
        if len(f) > 0:

            # First local minimum corresponds to initial contact
            thr = 0.20
            if (mid_foot_vel[ix_min_toe_vel_z[f[0]],0] / mid_foot_vel[ix_pk,0]) < thr:
                ix_IC.append(ix_min_toe_vel_z[f[0]])
        
        # Consider the positive peaks preceding the current (horizontal) peak
        f = np.argwhere(ix_max_heel_vel_z < ix_pk)[:,0]
        if len(f) > 0:

            # Last local maximum corresponds to final contact
            ix_FC.append(ix_max_heel_vel_z[f[-1]])
    
    # Correct initial contacts (and final contacts)
    ix_IC, ix_FC = np.array(ix_IC), np.array(ix_FC)
    return ix_IC, ix_FC

def _get_gait_events_OConnor(heel_pos, toe_pos, fs):
    """Detect gait events from optical motion capture data according to O'Connor et al. (2007).

    Parameters
    ----------
    heel_pos : (N, 3) array_like
        The heel marker position data.
    toe_pos : (N, 3) array_like
        The toe marker position data.
    fs : int, float
        Sampling frequency (in Hz).
    """

    # Calculate the mid foot
    mid_foot_pos = ( heel_pos + toe_pos ) / 2

    # Calculate the velocity signal
    mid_foot_vel = np.zeros_like(mid_foot_pos)
    mid_foot_vel[1:,:] = ( mid_foot_pos[1:,:] - mid_foot_pos[:-1,:] ) / (1./fs)
    mid_foot_vel[0,:] = mid_foot_vel[1,:]

    # Detect peaks in the velocity signals
    #   Define thresholds
    #       minimum horizontal velocity: 10% of the range in horizontal velocity
    #       minimum vertical velocity: 10% of the range in vertical veloctity
    #       minimum time between two successive peaks: 100 ms
    thr_min_vel_x = 0.1*(np.max(mid_foot_vel[:,0]) - np.min(mid_foot_vel[:,0]))
    thr_min_vel_z = 0.1*(np.max(mid_foot_vel[:,-1]) - np.min(mid_foot_vel[:,-1]))

    # Find (positive) peaks in the horizontal velocity
    ix_max_vel_x, _ = find_peaks(mid_foot_vel[:,0], height=thr_min_vel_x, distance=fs//4)

    # Find positive and negative peaks in the vertical velocity    
    ix_max_vel_z, _ = find_peaks(mid_foot_vel[:,-1], height=thr_min_vel_z, distance=fs//10)
    ix_min_vel_z, _ = find_peaks(-mid_foot_vel[:,-1], height=thr_min_vel_z, distance=fs//10)

    # For each peak in the horizontal velocity (assumed to correspond to midswing)
    ix_IC, ix_FC = [], []
    for ix_pk in ix_max_vel_x:

        # Consider the negative peaks following the current (horizontal) peak
        f = np.argwhere(np.logical_and(ix_min_vel_z > ix_pk, ix_min_vel_z < ix_pk+fs//2))[:,0]
        if len(f) > 0:

            # First local minimum corresponds to initial contact
            ix_IC.append(ix_min_vel_z[f[0]])
        
        # Consider the positive peaks preceding the current (horizontal) peak
        f = np.argwhere(ix_max_vel_z < ix_pk)[:,0]
        if len(f) > 0:

            # Last local maximum corresponds to final contact
            ix_FC.append(ix_max_vel_z[f[-1]])
    
    # Correct initial contacts (and final contacts)
    ix_IC, ix_FC = np.array(ix_IC), np.array(ix_FC)
    return ix_IC, ix_FC

def _get_gait_events_Zeni(heel_pos, toe_pos, pelvis_pos, fs):
    """Detect gait events from optical motion capture data according to Zeni Jr et al. (2008).

    Parameters
    ----------
    heel_pos : (N, 3) array_like
        The heel marker position data.
    toe_pos : (N, 3) array_like
        The toe marker position data.
    pelvis_pos : (N, 3) array_like
        The virtual pelvis marker position data.
    fs : int, float
        Sampling frequency (in Hz).
    """

    # Calculate marker position data relative to sacral markers
    heel_pos_rel = heel_pos - pelvis_pos
    toe_pos_rel = toe_pos - pelvis_pos

    # Subtract the mean
    heel_pos_rel = heel_pos_rel - np.mean(heel_pos_rel, axis=0)
    toe_pos_rel = toe_pos_rel - np.mean(toe_pos_rel, axis=0)

    # Set thresholds
    thr_min_height = 0.1 * ( np.max(heel_pos_rel[:,0]) - np.min(heel_pos_rel[:,0]) )
    thr_min_dist = fs//4

    # Find peaks in the heel marker relative position data
    ix_IC, _ = find_peaks(heel_pos_rel[:,0], height=thr_min_height, distance=thr_min_dist)

    # Find minima in the toe marker relative position data
    thr_min_height = 0.1 * ( np.max(toe_pos_rel[:,0]) - np.min(toe_pos_rel[:,0]) )
    ix_FC, _ = find_peaks(-toe_pos_rel[:,0], height=thr_min_height, distance=thr_min_dist)
    return np.array(ix_IC), np.array(ix_FC)

def _get_gait_events_from_OMC(data, fs, labels, method="OConnor"):
    """Detect gait events from optical motion capture (OMC) data according to the specified method.

    Parameters
    ----------
    data : (N, 3, M) array_like
        The marker position data with N time steps across 3 dimension for M markers.
    fs : int, float
        Sampling frequency (in Hz).
    labels : (M,) or (M,1) array_like
        The labels corresponding to the marker locations.
    method : str, optional
        The method used to detect the initial foot contacts and final contacts, by default "OConnor"
    """

    # Get position data for relevant markers
    l_heel_pos = _get_data_from_marker(data, labels, marker='l_heel')
    r_heel_pos = _get_data_from_marker(data, labels, marker='r_heel')
    l_toe_pos = _get_data_from_marker(data, labels, marker='l_toe')
    r_toe_pos = _get_data_from_marker(data, labels, marker='r_toe')
    l_psis_pos = _get_data_from_marker(data, labels, marker='l_psis')
    r_psis_pos = _get_data_from_marker(data, labels, marker='r_psis')
    l_asis_pos = _get_data_from_marker(data, labels, marker='l_asis')
    r_asis_pos = _get_data_from_marker(data, labels, marker='r_asis')
    
    # Calculate virtual pelvis marker
    pelvis_pos = ( l_psis_pos + r_psis_pos + l_asis_pos + r_asis_pos ) / 4

    # Switch methods  
    if method.upper() == "OCONNOR":
        # Left initial and final contacts
        l_ix_IC, l_ix_FC = _get_gait_events_OConnor(l_heel_pos, l_toe_pos, fs)

        # Right initial and final contacts
        r_ix_IC, r_ix_FC = _get_gait_events_OConnor(r_heel_pos, r_toe_pos, fs)
    elif method.upper() == "ZENI" or method.upper() == "ZENIJR":
        # Left initial and final contacts
        l_ix_IC, l_ix_FC = _get_gait_events_Zeni(l_heel_pos, l_toe_pos, pelvis_pos, fs)

        # Right initial and final contacts
        r_ix_IC, r_ix_FC = _get_gait_events_Zeni(r_heel_pos, r_toe_pos, pelvis_pos, fs)
    elif method.upper() == "PIJNAPPELS":
        # Left initial and final contacts
        l_ix_IC, l_ix_FC = _get_gait_events_Pijnappels(l_heel_pos, l_toe_pos, fs)

        # Right initial and final contacts
        r_ix_IC, r_ix_FC = _get_gait_events_Pijnappels(r_heel_pos, r_toe_pos, fs)
    else:
        pass
    return l_ix_IC, l_ix_FC, r_ix_IC, r_ix_FC

def _rotate_IMU_local_frame(acc, gyro, fs):
    """Rotate the IMU local frame such that:
        positive X axis points forward (anteroposterior direction)
        positive Y axis points to the left (mediolateral direction)
        positive Z axis points upward (vertical direction)

    Parameters
    ----------
    acc : (N, 3) array_like
        3D accelerometer data (in g) with N time steps across 3 dimensions.
    gyro : (N, 3) array_like
        3D gyroscope data (in degrees/s) with N time steps across 3 dimensions.
    fs : int, float
        Sampling frequency (in Hz).

    Returns
    -------
    acc_out, gyro_out : (N, 3) array_like
        3D accelerometer and 3D gyroscope data aligned with the anatomical axes.
    """

    acc_out = np.empty_like(acc)
    gyro_out = np.empty_like(gyro)

    # Determine the vertical axis from the mean accelerations
    ix_ax_VT = np.argmax(np.abs(np.mean(acc, axis=0)))

    # Determine the mediolateral axis from the energy of the gyroscope signals
    ix_ax_ML = np.argmax(np.sum(np.abs(gyro)**2, axis=0))

    # Remaining axis is the anteroposterior axis
    ix_ax_AP = np.setdiff1d(np.arange(3), np.array([ix_ax_ML, ix_ax_VT]))[0]

    # Determine whether mediolateral axis points to the left or right
    thr = 0.1 * ( np.max(gyro[:,ix_ax_ML]) - np.min(gyro[:,ix_ax_ML]) )
    ix_pks_pos, _ = find_peaks(gyro[:,ix_ax_ML], height=thr, distance=fs//4)
    ix_pks_neg, _ = find_peaks(-gyro[:,ix_ax_ML], height=thr, distance=fs//4)
    if np.percentile(np.abs(gyro[ix_pks_pos,ix_ax_ML]), 90) > np.percentile(np.abs(gyro[ix_pks_neg,ix_ax_ML]), 90):

        # Mediolateral axis points to the right
        if np.mean(acc[:,ix_ax_VT]) < -0.5:

            # Vertical axis points down, and thus anteroposterior axis points forward
            acc_out[:,0], gyro_out[:,0] = acc[:,ix_ax_AP], gyro[:,ix_ax_AP]
            acc_out[:,1], gyro_out[:,1] = -acc[:,ix_ax_ML], -gyro[:,ix_ax_ML]
            acc_out[:,2], gyro_out[:,2] = -acc[:,ix_ax_VT], -gyro[:,ix_ax_VT]
        
        elif np.mean(acc[:,ix_ax_VT]) > 0.5:

            # Vertical axis points up, and thus anteroposterior axis points backward
            acc_out[:,0], gyro_out[:,0] = -acc[:,ix_ax_AP], -gyro[:,ix_ax_AP]
            acc_out[:,1], gyro_out[:,1] = -acc[:,ix_ax_ML], -gyro[:,ix_ax_ML]
            acc_out[:,2], gyro_out[:,2] = acc[:,ix_ax_VT], gyro[:,ix_ax_VT]
        
        else:
            print("Not sure if vertical points up or down. Please check sensor alignment.")
            acc_out, gyro_out = acc, gyro
    else:

        # Mediolateral axis points to the left
        if np.mean(acc[:,ix_ax_VT]) < -0.5:

            # Vertical axis points down, and thus anteroposterior axis points backward
            acc_out[:,0], gyro_out[:,0] = -acc[:,ix_ax_AP], -gyro[:,ix_ax_AP]
            acc_out[:,1], gyro_out[:,1] = acc[:,ix_ax_ML], gyro[:,ix_ax_ML]
            acc_out[:,2], gyro_out[:,2] = -acc[:,ix_ax_VT], -gyro[:,ix_ax_VT]
        
        elif np.mean(acc[:,ix_ax_VT]) > 0.5:

            # Vertical axis points up, and thus anteroposterior axis points forward
            acc_out[:,0], gyro_out[:,0] = acc[:,ix_ax_AP], gyro[:,ix_ax_AP]
            acc_out[:,1], gyro_out[:,1] = acc[:,ix_ax_ML], gyro[:,ix_ax_ML]
            acc_out[:,2], gyro_out[:,2] = acc[:,ix_ax_VT], gyro[:,ix_ax_VT]
        
        else:
            print("Not sure if vertical points up or down. Please check sensor alignment.")
            acc_out, gyro_out = acc, gyro
    return acc_out, gyro_out

def _get_gait_events_Salarian(gyro_ML, fs):
    """Detect gait events from the mediolateral angular velocity of a shank-worn IMU
    following the methods from Salarian et al. (2004).

    Parameters
    ----------
    gyro_ML : (N, 1) array_like
        Mediolateral angular velocity (in degrees/s) with N time steps.
        It is assumed that the mediolateral direction is positive to the left, 
        thus we detect negative peaks corresponding to the midswing.
    fs : int, float
        Sampling frequency (in Hz).
    
    Returns
    -------
    ix_IC, ix_FC : array_like
        Arrays with the indexes corresponding to initial and final contacts, respectively.
    """

    # Remove drift
    filtered_gyro = _remove_drift_200Hz(gyro_ML)

    # Low-pass filter
    filtered_gyro = _butter_lowpass(filtered_gyro, fs)

    # Detect peaks corresponding to midswing
    thr = 50.0  # [...] peaks larger than 50 degrees/s were candidates for marking the midswing area [...]
    ix_MS, _ = find_peaks(-filtered_gyro, height=thr, distance=fs//2)

    # Detect peaks corresponding to initial contacts
    thr_ang_vel_IC = 10.0
    ix_pks, _ = find_peaks(gyro_ML, height=thr_ang_vel_IC)
    ix_IC = []
    for i in range(len(ix_MS)):
        # Find the nearest IC after midswing
        f = np.argwhere(np.logical_and(ix_pks > ix_MS[i], ix_pks <= ix_MS[i]+np.round(1.5*fs)))[:,0]
        if len(f) > 0:
            ix_IC.append(ix_pks[f[0]])
    
    # Detect peaks corresponding to final contacts
    thr_ang_vel_FC = 20.0
    ix_pks, _ = find_peaks(gyro_ML, height=thr_ang_vel_FC)
    ix_FC = []
    for i in range(len(ix_MS)):
        # Find the peak prior to midswing
        f = np.argwhere(np.logical_and(ix_pks < ix_MS[i], ix_pks >= ix_MS[i]-np.round(1.5*fs)))[:,0]
        if len(f) > 0:
            ix_FC.append(ix_pks[f[-1]])
    return np.array(ix_MS), np.array(ix_IC), np.array(ix_FC)

def _get_gait_events_from_IMU(imu_data, label=""):
    """Detect gait events from an inertial measurement unit (IMU) from the body position given by the label.

    Parameters
    ----------
    imu_data : dict
        Dictionary ... 
    fs : int, float
        Sampling frequency (in Hz).
    label : str, optional
        Label specifying the body position to which the IMU is attached, by default ""
    """
    # Get the accelerometer and gyroscope data
    acc = imu_data["acc"][:,:,np.argwhere(imu_data["imu_location"]==label)[:,0][0]]
    gyro = imu_data["gyro"][:,:,np.argwhere(imu_data["imu_location"]==label)[:,0][0]]
    fs = imu_data["fs"]

    if label == "":
        return
    elif ("ank" in label) or ("shank" in label):
        # Rotate the IMU local coordinate frame:
        # X in forward walking direction
        # Z in vertical upward direction
        # Y following from the right-hand rule (to the left)
        acc, gyro = _rotate_IMU_local_frame(acc, gyro, fs)

        # Detect events from mediolateral angular velocity
        ix_MS, ix_IC, ix_FC = _get_gait_events_Salarian(gyro[:,1], fs)
    return acc, gyro, ix_MS, ix_IC, ix_FC

