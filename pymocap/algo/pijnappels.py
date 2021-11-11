def pijnappels(heel_pos, toe_pos, fs, **kwargs):
    """Detects initial contacts and final contacts from the heel and toe marker position data.

    See:
        Pijnappels M, et al. Gait Posture. 2001. doi: 10.1016/s0966-6362(01)00110-2

    Parameters
    ----------
    heel_pos, toe_pos : (N, D) ndarray, (N, D) ndarray
        The marker position data with N time steps across D channels,
        for the heel and toe marker, respectively.
    fs : int, float
        The sampling frequency (in Hz).
    visualize : bool
        Boolean that indicates whether signals should be plotted.
        Defaults to False.

    Returns
    -------
    ix_IC, ix_FC : ndarray, ndarray
        The indexes corresponding to initial contacts and final contacts, respectively.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_prominences
    from pymocap.preprocessing import _butter_lowpass_filter

    # Get kwargs
    visualize = kwargs.get("visualize", False)

    # Lowpass filter
    heel_pos_filt = _butter_lowpass_filter(heel_pos, fs, 7.)
    toe_pos_filt  = _butter_lowpass_filter(toe_pos, fs, 7.)
    
    # Calculate virtual foot center 
    foot_center_pos = ( heel_pos_filt + toe_pos_filt ) / 2

    # Calculate corresponding velocity signals
    foot_center_vel = np.zeros_like(foot_center_pos)
    foot_center_vel[1:,:] = np.diff(foot_center_pos, axis=0) / (1/fs)
    foot_center_vel[0,:] = foot_center_vel[1,:]
        
    heel_vel = np.zeros_like(heel_pos_filt)
    heel_vel[1:,:] = np.diff(heel_pos_filt, axis=0) / (1/fs)
    heel_vel[0,:] = heel_vel[1,:]

    toe_vel = np.zeros_like(toe_pos_filt)
    toe_vel[1:,:] = np.diff(toe_pos_filt, axis=0) / (1/fs)
    toe_vel[0,:] = toe_vel[1,:]

    # Detect peaks in the foot center forward velocity
    ix_pks_x, _ = find_peaks(foot_center_vel[:,0], distance=fs//4)
    pk_proms = peak_prominences(foot_center_vel[:,0], ix_pks_x)
    ix_pks_x = ix_pks_x[pk_proms[0] > 0.1*max(pk_proms[0])]

    # Detect local minima in the toe vertical velocity
    thr = 0.1 * np.max(-toe_vel[:,2])  # local threshold
    ix_min_z_toe, _ = find_peaks(-toe_vel[:,2], height=thr)

    # Detect local maxima in the heel vertical velocity
    ix_max_z_heel, _ = find_peaks(heel_vel[:,2])

    ix_IC, ix_FC = [], []
    for i in range(len(ix_pks_x)):
        f = np.argwhere(np.logical_and(ix_min_z_toe > ix_pks_x[i], ix_min_z_toe <= ix_pks_x[i]+(fs+fs//2)))[:,0]
        if len(f) > 0:
            ix_IC.append(ix_min_z_toe[f[0]])
        g = np.argwhere(np.logical_and(ix_max_z_heel < ix_pks_x[i], ix_max_z_heel >= ix_pks_x[i]-(fs+fs//2)))[:,0]
        if len(g) > 0:
            ix_FC.append(ix_max_z_heel[g[-1]])
    
    # Visualize
    if visualize == True:
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(heel_pos_filt[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.4))
        axs[0].plot(toe_pos_filt[:,2], ls='-.', lw=1, c=(0, 0.5, 0, 0.2))
        axs[0].plot(ix_IC, heel_pos_filt[ix_IC,2], 'o', mfc='none', mec=(0, 0.5, 0), ms=8)
        axs[0].set_xlim((0, toe_pos_filt.shape[0]))

        axs[1].plot(heel_vel[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.4))
        axs[1].plot(toe_vel[:,2], ls='-', lw=1, c=(0, 0.5, 0, 0.2))
        axs[1].plot(ix_IC, toe_vel[ix_IC,2], 'o', mfc='none', mec=(0, 0.5, 0), ms=8)
        axs[1].plot(ix_FC, heel_vel[ix_FC,2], 's', mfc='none', mec=(0, 0.5, 0), ms=8)
        axs[1].set_xlim((0, toe_vel.shape[0]))

        axs[2].plot(foot_center_vel[:,0], ls='-', lw=1, c=(0, 0.5, 0))
        axs[2].plot(ix_pks_x, foot_center_vel[ix_pks_x,0], '*', c=(0, 0.5, 0))
        for i in range(len(ix_pks_x)):
            axs[0].plot([ix_pks_x[i], ix_pks_x[i]], [np.min(toe_pos_filt[:,2]), np.max(heel_pos_filt[:,2])], ls='-', lw=1, c=(1, 0.5, 0))
            axs[1].plot([ix_pks_x[i], ix_pks_x[i]], [np.min(heel_vel[:,2]), np.max(heel_vel[:,2])], ls='-', lw=1, c=(1, 0.5, 0))
            axs[2].plot([ix_pks_x[i], ix_pks_x[i]], [np.min(foot_center_vel[:,0]), np.max(foot_center_vel[:,0])], ls='-', lw=1, c=(1, 0.5, 0))
        axs[2].set_xlim((0, foot_center_vel.shape[0]))
        plt.show()
    return np.array(ix_IC), np.array(ix_FC)