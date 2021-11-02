def _butter_lowpass(N, Wn):
    """Designs a Butterworth lowpass filter.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
    """
    from scipy.signal import butter
    b, a = butter(N, Wn, btype="low")
    return b, a

def _butter_lowpass_filter(data, fs, fc, **kwargs):
    """Lowpass filters the data using a Butterworth filter.
    
    Parameters
    ----------
    data : (N, D) ndarray
        The input data with N time steps across D channels.
    fs : int, float
        The sampling frequency (in Hz).
    fc : int, float
        The cut-off frequency (in Hz).
    forder : int, optional
        The order of the filter, defaults to 4.
    
    Returns
    -------
    filtered_data : (N, D) ndarray
        The lowpass filtered data.
    """
    from scipy.signal import filtfilt

    # Parse optional args
    forder = kwargs.get("forder", 4)

    # Get filter polynomials
    b, a = _butter_lowpass(forder, fc/(fs/2))

    # Filter data along axis=0
    filtered_data = filtfilt(b, a, data, axis=0, padtype="odd", padlen=3*(max(len(b), len(a))-1))
    return filtered_data

def _remove_drift(data):
    """Removes drift from the data by highpass filtering.

    Parameters
    ----------
    data : (N, D) ndarray
        The input data with N time steps across D channels.
    
    Returns
    -------
    filtered_data : (N, D) ndarray)
        The highpass filtered data.
    """
    from scipy.signal import filtfilt

    # Get filter polynomials
    b = [1., -1.]
    a = [1., -0.995]

    # Filter data along axis=0
    filtered_data = filtfilt(b, a, data, axis=0, padtype="odd", padlen=3*(max(len(b), len(a))-1))
    return filtered_data

def _resample_data(data, fs_old, fs_new):
    """Resamples data.

    Parameters
    ----------
    data : (N, D) ndarray
        The input data.
    fs_old : int, float
        The original sampling frequency (in Hz).
    fs_new : int, float
        The desired sampling frequency (in Hz).
    
    Returns
    -------
    resampled_data : (N', D) ndarray
        The resampled data.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    # Make sure that D axis exists
    try:
        N, D = data.shape
    except:
        data = np.reshape(data, (data.shape[0], 1))
        N, D = data.shape
    
    # Original timestamps
    t_old = np.arange(N) / fs_old
    t_old = t_old[np.logical_not(np.any(np.isnan(data), axis=1))]
    data = data[np.logical_not(np.any(np.isnan(data), axis=1)),:]

    # Number of time steps after interpolation
    N_prime = int(t_old[-1]/(1/fs_new))+1
    t_new = np.arange(N_prime)/fs_new

    # Allocate memory for output variable
    resampled_data = np.zeros((N_prime, D))

    # Loop over the channels
    for d in range(D):
        f = interp1d(t_old, data[:,d])
        resampled_data[:,d] = f(t_new)
    return resampled_data