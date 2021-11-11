def salarian(x, fs, **kwargs):
    """Detects initial contacts and final contacts from the mediolateral angular velocity of a shank-worn IMU.

    See:
        Salarian A, et al. IEEE Trans Biomed Eng. 2004. doi: 10.1109/TBME.2004.827933 

    Parameters
    ----------
    x : (N, D) array
        The input data with N time steps across D channels.
    fs : int, float
        The sampling frequency (in Hz).
    thr_MS : float, optional
        The minimum angular velocity (in deg/s) for a relevant peak associated with midswing.
        Defaults to 50.0 degrees/s.
    thr_IC : float, optional
        The angular velocity (in deg/s) for a relevant peak associated with initial contact.
        Defaults to 10.0 degrees/s.
    thr_FC : float, optional
        The angular velocity (in deg/s) for a relevant peak associated with final contact.
        Defaults to 20.0 degrees/s.

    Returns
    -------
    ix_IC, ix_FC : ndarray, ndarray
        The indexes corresponding to initial contacts and final contacts, respectively.
    """
    import numpy as np
    from scipy.signal import find_peaks
    from pymocap.preprocessing import _remove_drift, _butter_lowpass_filter

    # Get kwargs
    visualize = kwargs.get("visualize", False)
    thr_MS = kwargs.get("thr_MS", 50.0)
    thr_IC = kwargs.get("thr_IC", 10.0)
    thr_FC = kwargs.get("thr_FC", 20.0)

    # Remove drift
    x_filt = _remove_drift(x)

    # Lowpass filter
    x_filt_filt = _butter_lowpass_filter(x_filt, fs, 3.)

    # Detect peaks associated with midswings
    ix_MS, _ = find_peaks(x_filt_filt, height=thr_MS, distance=fs//2)
    ix_MS = ix_MS[x[ix_MS] > thr_MS]

    # Detect peaks associated with initial contact
    ix_pks_IC, _ = find_peaks(-x_filt, height=thr_IC)
    ix_IC = []
    for i in range(len(ix_MS)):
        f = np.argwhere(np.logical_and(ix_pks_IC > ix_MS[i], ix_pks_IC<ix_MS[i]+(fs+fs//2)))[:,0]
        if len(f)>0:
            ix_IC.append(ix_pks_IC[f[0]])
    
    # Detect peaks associated with final contact
    ix_pks_FC, _ = find_peaks(-x_filt, height=thr_FC)
    ix_FC = []
    for i in range(len(ix_MS)):
        f = np.argwhere(np.logical_and(ix_pks_FC < ix_MS[i], ix_pks_FC > ix_MS[i]-(fs+fs//2)))[:,0]
        if len(f)>0:
            ix_FC.append(ix_pks_FC[f[-1]])
    return np.array(ix_IC), np.array(ix_FC)