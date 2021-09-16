from scipy.io.matlab.mio5_utils import squeeze_element

def _load_file(filename):
    """Loads data from a data file organized in a MATLAB struct.

    See:
        https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python
        
    Parameters
    ----------
    filename : str
        Filename of the datafile. Currently only *.mat files are supported.
    """
    import scipy.io as sio

    def _check_keys(dict):
        """Checks if entries in a dictionary are mat objects.
        If so, then to_dict() is called to convert them to nested dictionaries.

        Parameters
        ----------
        dict : dict
            Input dictionary.
        """
        for key in dict:
            if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
                dict[key] = _to_dict(dict[key])
        return dict
    
    def _to_dict(mat_obj):
        """Converts a mat object to a nested dictionary.
        
        Parameters
        ----------
        mat_obj : mat_obj
            The input mat object.

        Returns
        -------
        dict : dict
            The nested dictionary.
        """
        dict = {}
        for strg in mat_obj._fieldnames:
            elem = mat_obj.__dict__[strg]
            if isinstance(elem, sio.matlab.mio5_params.mat_struct):
                dict[strg] = _to_dict(elem)
            else:
                dict[strg] = elem
        return dict
    
    # Naive import of data
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Call recursive function to get nested dictionary
    d = _check_keys(data)

    # Return data dictionary
    return d["data"]