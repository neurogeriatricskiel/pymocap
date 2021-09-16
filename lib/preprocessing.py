import numpy as np
from numpy.matlib import repmat
import warnings


def _predict_missing_markers(data_gaps, **kwargs):
    """Fills gaps in the marker postion data exploiting intercorrelations between marker coordinates.

    See:


    Parameters
    ----------
    data_gaps : (N, M) array_like
        Array of marker position data with N time steps across M channels.
        The data need to be organized as follows:

        x1(t1) y1(t1) z1(t1) x2(t1) y2(t1) z2(t1) ...    xm(t1) ym(t1) zm(t1)
        x1(t2) y1(t2) z1(t2) x2(t2) y2(t2) z2(t2) ...    xm(t2) ym(t2) zm(t2)
        ...    ...    ...    ...    ...    ...    ...    ...    ...    ...
        x1(tn) y1(tn) z1(tn) x2(tn) y2(tn) z2(tn) ...    xm(tn) ym(tn) zm(tn)
    
    Optional parameters
    -------------------
        method : str
            Reconstruction strategy for gaps in multiple markers (`R1` or `R2`).
        weight_scale : float
            Parameter `sigma` for determining weights. Default is 200.
        mm_weight : float
            Parapmeter for weight on missing markers. Default is 0.02.
        distal_threshold : float
            Cut-off distance for distal marker in `R2` relative to average Euclidean distances. Default is 0.5.
        min_cum_sv : float
            Minimum cumulative sum of eigenvalues of the normalized singular values.
            Determines the number of principal component vectors included in the analysis. Default is 0.99.
    """
    def _distance2marker(data, ix_channels_with_gaps):
        """Computes the Euclidean distance for each marker with missing data to each other marker.

        Parameters
        ----------
        data : (N, M) array_like
            The marker position data with N time steps across M channels.
        ix_channels_with_gaps : array_lik
            The indexes of the channels with missing marker data.
        """

    def _reconstruct(data, **kwargs):
        """Reconstructs missing marker data using the strategy based on intercorrelations between marker clusters.

        Parameters
        ----------
        data : (N, M) array_like
            Array of marker position data with N time steps across M channels.
            Note that the mean trajectories have been subtracted from the original marker position data,
            as to obtain a coordinate system moving with the subject.

        Returns
        -------
        [type]
            [description]
        """

        # Get shape of data
        N, M = data.shape

        # Find channels with missing data
        ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data), axis=0))

        # Compute the weights
        weights = _distance2marker(data, ix_channels_with_gaps)
        return weights

    # Parse optional arguments, or get defaults
    method = kwargs.get("method", "R2")
    weight_scale = kwargs.get("weight_scale", 200)
    mm_weight = kwargs.get("mm_weight", 0.02)
    distal_threshold = kwargs.get("distal_threshold", 0.5)
    min_cum_sv = kwargs.get("min_cum_sv", 0.99)

    # Get number of time steps, number of channels, and corresponding number of markers
    n_time_steps, n_channels = data_gaps.shape
    n_markers = n_channels // 3

    # Find channels with gaps
    ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=0))
    
    # Find time steps with gaps
    ix_time_steps_with_gaps, = np.nonzero(np.any(np.isnan(data_gaps), axis=1))
    
    # If no gaps were found
    if len(ix_time_steps_with_gaps) == 0:
        warnings.warn("Submitted data appear to have no gaps. Make sure that gaps missing data is represented by NaNs.")
        return data_gaps
    
    # Subtract mean marker trajectory to get a coordinate system moving with the subject
    T = np.delete(data_gaps, ix_channels_with_gaps, axis=1)
    mean_trajectory_x = np.mean(T[:,::3], axis=1)
    mean_trajectory_y = np.mean(T[:,1::3], axis=1)
    mean_trajectory_z = np.mean(T[:,2::3], axis=1)
    del T

    B = data_gaps.copy()
    B[:, ::3] = B[:, ::3] - np.tile(mean_trajectory_x.reshape(-1,1), (1, n_markers))
    B[:, 1::3] = B[:, 1::3] - np.tile(mean_trajectory_y.reshape(-1,1), (1, n_markers))
    B[:, 2::3] = B[:, 2::3] - np.tile(mean_trajectory_z.reshape(-1,1), (1, n_markers))

    # Reconstruct missing marker data
    if method == "R1":
        reconstructed_data = _reconstruct(B)

    
    return reconstructed_data