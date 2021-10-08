from math import dist
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance

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
    mid_foot_vel = ( mid_foot_pos[1:,:] - mid_foot_pos[:-1,:] ) / (1./fs)

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
        f = np.argwhere(np.logical_and(ix_min_vel_z > ix_pk, ix_min_vel_z < ix_pk+fs//4))[:,0]
        if len(f) > 0:

            # First local minimum corresponds to initial contact
            ix_IC.append(ix_min_vel_z[f[0]])
        
        # Consider the positive peaks preceding the current (horizontal) peak
        f = np.argwhere(ix_max_vel_z < ix_pk)[:,0]
        if len(f) > 0:

            # Last local maximum corresponds to final contact
            ix_FC.append(ix_max_vel_z[f[-1]])
    return np.array(ix_IC), np.array(ix_FC)

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
    l_heel_pos = np.squeeze(data[:,:,np.argwhere(labels=='l_heel')[:,0]], axis=-1)
    r_heel_pos = np.squeeze(data[:,:,np.argwhere(labels=='r_heel')[:,0]], axis=-1)
    l_toe_pos = np.squeeze(data[:,:,np.argwhere(labels=='l_toe')[:,0]], axis=-1)
    r_toe_pos = np.squeeze(data[:,:,np.argwhere(labels=='r_toe')[:,0]], axis=-1)
    l_psis_pos = np.squeeze(data[:,:,np.argwhere(labels=='l_psis')[:,0]], axis=-1)
    r_psis_pos = np.squeeze(data[:,:,np.argwhere(labels=='r_psis')[:,0]], axis=-1)
    l_asis_pos = np.squeeze(data[:,:,np.argwhere(labels=='l_asis')[:,0]], axis=-1)
    r_asis_pos = np.squeeze(data[:,:,np.argwhere(labels=='r_asis')[:,0]], axis=-1)
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
    else:
        pass
    return l_ix_IC, l_ix_FC, r_ix_IC, r_ix_FC