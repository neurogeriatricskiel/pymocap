import numpy as np
from numpy.core.fromnumeric import std
from numpy.matlib import repmat
import warnings


def _predict_missing_markers(data_gaps, **kwargs):
    """Fills gaps in the marker postion data exploiting intercorrelations between marker coordinates.

    See:
        Federolf PA (2013), PLoS ONE 8(10):e78689. doi:10.1371/journal.pone.0078689
        Gløersen Ø, Federolf P (2016), PLoS ONE 11(3):e0152616. doi:10.1371/journal.pone.0152616

    Parameters
    ----------
    data_gaps : (N, M) array_like
        Array of marker position data with N time steps across M channels.
        The data need to be organized as follows:

        x1(t1) y1(t1) z1(t1) x2(t1) y2(t1) z2(t1) ...    xm(t1) ym(t1) zm(t1)
        x1(t2) y1(t2) z1(t2) x2(t2) y2(t2) z2(t2) ...    xm(t2) ym(t2) zm(t2)
        ...    ...    ...    ...    ...    ...    ...    ...    ...    ...
        x1(tn) y1(tn) z1(tn) x2(tn) y2(tn) z2(tn) ...    xm(tn) ym(tn) zm(tn)

        Thus, the first three columns correspond to the x-, y-, and-z coordinate of the 1st marker.
        The rows correspond to the consecutive time steps (i.e., frames).
    
    Optional parameters
    -------------------
        method : str
            Reconstruction strategy for gaps in multiple markers (`R1` or `R2` (default)).
        weight_scale : float
            Parameter `sigma` for determining weights. Default is 200.
        mm_weight : float
            Parameter for weight on missing markers. Default is 0.02.
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
        
        Returns
        -------
        weights : (M'', M') array_like
            Array of pair-wise Euclidean distances between markers with gaps and each other marker.
            Here, M'' is the number of markers with missing data, and M' is the number of markers.
        """
        from scipy.spatial.distance import cdist

        # Get shape of data
        N, M = data.shape
        
        # Reshape data to shape (3, n_markers, n_time_steps)
        ix_markers_with_gaps = ( ix_channels_with_gaps[2::3] // 3 )  # columns of markers with gaps
        n_markers_with_gaps = len(ix_markers_with_gaps)
        data_reshaped = (data.T).reshape((3, M//3, N), order="F")

        # Compute weights based on distances
        weights = np.empty((n_markers_with_gaps, M//3, N))
        for i in range(N):
            weights[:,:,i] = cdist(data_reshaped[:,ix_markers_with_gaps,i].T, data_reshaped[:,:,i].T, "euclidean")
        weights = np.nanmean(weights, axis=-1)
        return weights
    
    def _PCA(data):
        """Performs principal components analysis by means of singular value decomposition.

        Parameters
        ----------
        data : (N, M) array_like
            The marker position data with N time steps across M channels.
        
        Returns
        -------
        PC : 
            The principal component vectors.
        sqrtEV : 
            The square root of the eigenvalues.
        """

        # Get shape of data
        N, M = data.shape

        # Calculate Y matrix
        Y = data / np.sqrt(N-1)

        # Find principal components
        _, sqrtEV, VT = np.linalg.svd(Y, full_matrices=0)
        PC = VT.T
        return PC, sqrtEV

    def _reconstruct(data, weight_scale, mm_weight, min_cum_sv):
        """Reconstructs missing marker data using the strategy based on intercorrelations between marker clusters.

        Parameters
        ----------
        data : (N, M) array_like
            Array of marker position data with N time steps across M channels.
            Note that the mean trajectories have been subtracted from the original marker position data,
            as to obtain a coordinate system moving with the subject.

        Returns
        -------
        reconstructed_data : (N, M) array_like
            The data with reconstructed marker trajectories.
        """

        # Get shape of data
        n_time_steps, n_channels = data.shape

        # Find channels with missing data
        ix_channels_with_gaps, = np.nonzero(np.any(np.isnan(data), axis=0))
        ix_time_steps_with_gaps, = np.nonzero(np.any(np.isnan(data), axis=1))

        # Compute the weights
        weights = _distance2marker(data, ix_channels_with_gaps)
        if weights.shape[0] >= 1:
            weights = np.min(weights, axis=0)
        print(f"Shape of weights vector: {weights.shape}")
        weights = np.exp(-np.divide(weights**2, 2*weight_scale**2))
        weights[ix_channels_with_gaps[2::3]//3] = mm_weight

        # Define matrices need for reconstruction
        M_zeros = data.copy()
        M_zeros[:,ix_channels_with_gaps] = 0
        N_no_gaps = np.delete(data, ix_time_steps_with_gaps, axis=0)
        N_zeros = N_no_gaps.copy()
        N_zeros[:,ix_channels_with_gaps] = 0

        # Normalize matrices to unit variance, then multiply by weights
        mean_N_no_gaps = np.mean(N_no_gaps, axis=0)
        mean_N_zeros = np.mean(N_zeros, axis=0)
        stdev_N_no_gaps = np.std(N_no_gaps, axis=0)
        stdev_N_no_gaps[np.argwhere(stdev_N_no_gaps==0)[:,0]] = 1

        M_zeros = np.divide(( M_zeros - np.tile(mean_N_zeros.reshape(-1,1).T, (M_zeros.shape[0], 1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (M_zeros.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (M_zeros.shape[0],1))
        
        N_no_gaps = np.divide(( N_no_gaps - np.tile(mean_N_no_gaps.reshape(-1,1).T, (N_no_gaps.shape[0],1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (N_no_gaps.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (N_no_gaps.shape[0],1))
        
        N_zeros = np.divide(( N_zeros - np.tile(mean_N_zeros.reshape(-1,1).T, (N_zeros.shape[0],1)) ), \
            np.tile(stdev_N_no_gaps.reshape(-1,1).T, (N_zeros.shape[0],1))) * \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (N_zeros.shape[0],1))

        # Calculate the principal component vectors and the eigenvalues
        PC_vectors_no_gaps, sqrtEV_no_gaps = _PCA(N_no_gaps)
        PC_vectors_zeros, sqrtEV_zeros = _PCA(N_zeros)

        # Determine the number of eigenvectors to consider
        n_eigvecs = np.max([np.argwhere(np.cumsum(sqrtEV_no_gaps) >= min_cum_sv*np.sum(sqrtEV_no_gaps))[:,0][0], \
            np.argwhere(np.cumsum(sqrtEV_zeros) >= min_cum_sv*np.sum(sqrtEV_zeros))[:,0][0]])
        PC_vectors_no_gaps = PC_vectors_no_gaps[:,:n_eigvecs+1]
        PC_vectors_zeros = PC_vectors_zeros[:,:n_eigvecs+1]

        # Calculate the transformation matrix
        T = PC_vectors_no_gaps.T @ PC_vectors_zeros

        # Calculate the reconstruction matrix, see: Federolf (2013).
        R = M_zeros @ PC_vectors_zeros @ T @ PC_vectors_no_gaps.T

        # Reverse the normalization
        R = np.tile(mean_N_no_gaps.reshape(-1,1).T, (data.shape[0],1)) + \
            np.divide(R * np.tile(stdev_N_no_gaps.reshape(-1,1).T, (data.shape[0],1)), \
                np.tile(np.tile(weights, (3,1)).reshape(n_channels, order="F").reshape(-1,1).T, (data.shape[0],1)))
        
        # Replace missing data with reconstructed data
        reconstructed_data = data.copy()
        for ix in ix_channels_with_gaps:
            reconstructed_data[:,ix] = R[:,ix]
        return reconstructed_data

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
        warnings.warn("Submitted data appear to have no gaps. Make sure that gaps are represented by NaNs.")
        return data_gaps
    elif len(ix_time_steps_with_gaps) == n_time_steps:
        if method == "R1":
            warnings.warn("For each time step there is at least one marker with missing data. Cannot perform reconstruction according to strategy R1.")
            return data_gaps
    
    # Subtract mean marker trajectory to get a coordinate system moving with the subject
    T = np.delete(data_gaps, ix_channels_with_gaps, axis=1)
    mean_trajectory_x = np.mean(T[:,::3], axis=1)
    mean_trajectory_y = np.mean(T[:,1::3], axis=1)
    mean_trajectory_z = np.mean(T[:,2::3], axis=1)
    del T

    B = data_gaps.copy()
    B[:,::3] = B[:,::3] - np.tile(mean_trajectory_x.reshape(-1,1), (1, n_markers))
    B[:,1::3] = B[:,1::3] - np.tile(mean_trajectory_y.reshape(-1,1), (1, n_markers))
    B[:,2::3] = B[:,2::3] - np.tile(mean_trajectory_z.reshape(-1,1), (1, n_markers))

    # Reconstruct missing marker data
    if method == "R1":
        reconstructed_data = _reconstruct(B, weight_scale=weight_scale, mm_weight=mm_weight, min_cum_sv=min_cum_sv)

        # Replace the missing data with reconstructed data
        filled_data = np.where(np.isnan(data_gaps), reconstructed_data, B)
    elif method == "R2":
        # Allocate space for reconstructed data matrix
        reconstructed_data = B.copy()

        # Get markers with gaps
        ix_markers_with_gaps = ix_channels_with_gaps[2::3] // 3
        for ix in ix_markers_with_gaps[:1]:
            eucl_distance_2_markers = _distance2marker(B, np.arange(ix*3,(ix+1)*3))
            thresh = distal_threshold * np.mean(eucl_distance_2_markers)
            ix_channels_2_zero = np.argwhere(np.logical_and(np.reshape(np.tile(eucl_distance_2_markers, (3,1)), (1,111), order="F").reshape(-1,) > thresh, \
                np.any(np.isnan(B), axis=0)))[:,0]
            
            # Set channels to 0, for which there are NaNs and that are far away from the current marker
            data_gaps_removed_cols = B.copy()
            data_gaps_removed_cols[:,ix_channels_2_zero] = 0
            data_gaps_removed_cols[:,ix*3:(ix+1)*3] = B[:,ix*3:(ix+1)*3]

            # Find gaps in marker trajectory
            ix_frames_with_gaps, = np.nonzero(np.isnan(B[:, ix]))
            
            # For channels that have gaps in the same time span, set values to 0
            for jx in np.setdiff1d(ix_markers_with_gaps, ix):
                if np.any(np.isnan(data_gaps_removed_cols[ix_frames_with_gaps,3*jx])):
                    data_gaps_removed_cols[:,3*jx:3*jx+3] = 0
            
            # Find frames without gaps in marker trajectory
            ix_frames_no_gaps, = np.nonzero(np.logical_not(np.any(np.isnan(data_gaps_removed_cols), axis=1)))

            # Find frames with gaps in marker trajectory `ix`
            ix_frames_2_reconstruct, = np.nonzero(np.any(np.isnan(data_gaps_removed_cols[:,3*ix:3*ix+3]), axis=1))

            # Concatenate frames to reconstruct, at the end frames without gaps
            ix_complete_and_gapped_frames = np.concatenate((ix_frames_no_gaps, ix_frames_2_reconstruct))

            # Get indexes of frames to fill
            ix_fill_frames = np.arange(len(ix_frames_no_gaps), len(ix_complete_and_gapped_frames))

            # Store temporarily reconstruct data
            temp_reconstructed_data = _reconstruct(data_gaps_removed_cols[ix_complete_and_gapped_frames,:], weight_scale=weight_scale, mm_weight=mm_weight, min_cum_sv=min_cum_sv)

            # Replace gapped data with reconstructed data
            reconstructed_data[ix_frames_2_reconstruct, 3*ix:3*ix+3] = temp_reconstructed_data[ix_fill_frames, 3*ix:3*ix+3]
        
        # Assign to output variable
        filled_data = reconstructed_data.copy()

    else:
        warnings.warn("Invalid reconstruction method, please specify `R1` or `R2`. Returning original data.")
        return data_gaps

    # Add the mean marker trajectory
    filled_data[:,::3] = filled_data[:,::3] + np.tile(mean_trajectory_x.reshape(-1,1), (1,n_markers))
    filled_data[:,1::3] = filled_data[:,1::3] + np.tile(mean_trajectory_y.reshape(-1,1), (1,n_markers))
    filled_data[:,2::3] = filled_data[:,2::3] + np.tile(mean_trajectory_z.reshape(-1,1), (1,n_markers))
    return filled_data