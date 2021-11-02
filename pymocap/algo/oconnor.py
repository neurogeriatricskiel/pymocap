def oconnor(heel_pos, toe_pos, fs, **kwargs):
    """Detects initial contacts and final contacts from the heel and toe marker position data.

    See:
        O'Connor CM, et al. Gait Posture. 2007. doi: 10.1016/j.gaitpost.2006.05.016 

    Parameters
    ----------
    heel_pos, toe_pos : (N, D) ndarray, (N, D) ndarray
        The marker position data with N time steps across D channels,
        for the heel and toe marker, respectively.
    fs : int, float
        The sampling frequency (in Hz).

    Returns
    -------
    ix_IC, ix_FC : ndarray, ndarray
        The indexes corresponding to initial contacts and final contacts, respectively.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_prominences
    from pymocap.preprocessing import _butter_lowpass_filter

    # Lowpass filter
    heel_pos_filt = _butter_lowpass_filter(heel_pos, fs, 7.)
    toe_pos_filt  = _butter_lowpass_filter(toe_pos, fs, 7.)

    # Calculate virtual foot center 
    foot_center_pos = ( heel_pos_filt + toe_pos_filt ) / 2

    # Calculate the velocity
    foot_center_vel = np.zeros_like(foot_center_pos)
    foot_center_vel[1:,:] = np.diff(foot_center_pos, axis=0) / (1/fs)
    foot_center_vel[0,:] = foot_center_vel[1,:]

    # Detect peaks in the foot center forward velocity
    ix_pks_x, _ = find_peaks(foot_center_vel[:,0], distance=fs//4)
    pk_proms = peak_prominences(foot_center_vel[:,0], ix_pks_x)
    ix_pks_x = ix_pks_x[pk_proms[0] > 0.1*max(pk_proms[0])]

    # Detect negative peaks in the foot center vertical velocity
    ix_pks_z_neg, _ = find_peaks(-foot_center_vel[:,2])
    ix_pks_z_pos, _ = find_peaks(foot_center_vel[:,2])

    ix_IC, ix_FC = [], []
    for i in range(len(ix_pks_x)):
        f = np.argwhere(np.logical_and(ix_pks_z_neg > ix_pks_x[i], ix_pks_z_neg <= ix_pks_x[i]+(fs+fs//2)))[:,0]
        if len(f) > 0:
            ix_IC.append(ix_pks_z_neg[f[0]])
        g = np.argwhere(np.logical_and(ix_pks_z_pos < ix_pks_x[i], ix_pks_z_pos >= ix_pks_x[i]-(fs+fs//2)))[:,0]
        if len(g) > 0:
            ix_FC.append(ix_pks_z_pos[g[-1]])
    
    # Visualize
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(heel_pos[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.4))
    # ax[0].plot(toe_pos[:,2], ls='-.', lw=1, c=(0, 0.5, 0, 0.2))
    # ax[0].plot(foot_center_pos[:,2], ls='-', lw=1, c=(0, 0.5, 0))
    # ax[0].set_xlim((0, foot_center_pos.shape[0]))

    # ax[1].plot(foot_center_vel[:,2], ls='-', lw=1, c=(0, 0.5, 0))
    # ax[1].plot(ix_IC, foot_center_vel[ix_IC,2], 'o', mfc='none', mec=(0, 0.5, 0), ms=8)
    # ax[1].plot(ix_FC, foot_center_vel[ix_FC,2], 's', mfc='none', mec=(0, 0.5, 0), ms=8)
    # ax[1].set_xlim((0, foot_center_vel.shape[0]))
    
    # ax[2].plot(foot_center_vel[:,0], ls='-', lw=1, c=(0, 0.5, 0))
    # ax[2].plot(ix_pks_x, foot_center_vel[ix_pks_x,0], '^', mfc='none', mec=(0, 0.5, 0), ms=4)
    # ax[2].set_xlim((0, foot_center_vel.shape[0]))
    # plt.show()
    return np.array(ix_IC), np.array(ix_FC)