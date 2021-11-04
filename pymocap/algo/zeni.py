def zeni(heel_pos, toe_pos, psis_pos, fs, **kwargs):
    """Detects initial contacts and final contacts from the heel and toe marker position relative to the psis marker.

    See:
        Zeni Jr, et al. Gait Posture. 2008. doi: 10.1016/j.gaitpost.2007.07.007

    Parameters
    ----------
    heel_pos, toe_pos, psis_pos : (N, D) ndarray, (N, D) ndarray, (N, D) ndarray
        The marker position data with N time steps across D channels,
        for the heel, toe and posterior iliac spine marker, respectively.
    fs : int, float
        The sampling frequency (in Hz).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, peak_prominences
    from pymocap.preprocessing import _butter_lowpass_filter

    # Lowpass filter
    heel_pos_filt = _butter_lowpass_filter(heel_pos, fs, fc=7.)
    toe_pos_filt  = _butter_lowpass_filter(toe_pos, fs, fc=7.)
    psis_pos_filt = _butter_lowpass_filter(psis_pos, fs, fc=7.)

    # Calculate position relative to iliac spine marker
    heel_pos_rel = heel_pos_filt - psis_pos_filt
    toe_pos_rel  = toe_pos_filt - psis_pos_filt

    # Find local maxima in relative heel position
    ix_max_x_heel, _ = find_peaks(heel_pos_rel[:,0], distance=fs//4)
    pk_proms = peak_prominences(heel_pos_rel[:,0], ix_max_x_heel)
    ix_max_x_heel = ix_max_x_heel[pk_proms[0] > 0.1*max(pk_proms[0])]

    # Find local minima in relative toe position
    ix_min_x_toe, _ = find_peaks(-toe_pos_rel[:,0], distance=fs//4)
    pk_proms = peak_prominences(-toe_pos_rel[:,0], ix_min_x_toe)
    ix_min_x_toe = ix_min_x_toe[pk_proms[0] > 0.1*max(pk_proms[0])]

    # Assign output variables
    ix_IC, ix_FC = ix_max_x_heel, ix_min_x_toe

    # Visualize
    # fig, ax = plt.subplots(1, 1, figsize=(21., 14.8))
    # ax.plot(heel_pos_rel[:,0], ls='-', lw=1, c=(0, 0.5, 0))
    # ax.plot(ix_max_x_heel, heel_pos_rel[ix_max_x_heel,0], 'o', mfc='none', mec=(0, 0.5, 0))
    # ax.plot(toe_pos_rel[:,0], ls='-', lw=1, c=(0, 0.5, 0, 0.5))
    # ax.plot(ix_min_x_toe, toe_pos_rel[ix_min_x_toe,0], 'o', mfc='none', mec=(0, 0.5, 0, 0.5))
    # ax.set_xlim((0, heel_pos_rel.shape[0]))
    # plt.show()
    return ix_IC, ix_FC