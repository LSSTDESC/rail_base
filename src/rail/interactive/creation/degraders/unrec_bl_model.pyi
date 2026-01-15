from typing import Any

def unrec_bl_model(**kwargs) -> Any:
    """
    Model for Creating Unrecognized Blends.

    Finding objects nearby each other. Merge them into one blended
    Use Friends of Friends for matching. May implement shape matching in the future.
    Take avergaged Ra and Dec for blended source, and sum up fluxes in each band. May
    implement merged shapes in the future.

    Requires gcc, which depending on your installation, may be difficult for the caller
    (FoFCatalogMatching dependency fast3tree) to find. Conda-installed gcc seems to fix
    this.

    ---

    The main interface method for ``Degrader``.

    Applies degradation.

    This will attach the sample to this `Degrader` (for introspection and
    provenance tracking).

    Then it will call the run() and finalize() methods, which need to be
    implemented by the sub-classes.

    The run() method will need to register the data that it creates to this
    Estimator by using ``self.add_data('output', output_data)``.

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.unrec_bl_model.UnrecBlModel.__call__

    Parameters
    ----------
    input : table-like
        The sample to be degraded
    seed : int, default=None
        An integer to set the numpy random seed
    ra_label : str, optional
        ra column name
        Default: ra
    dec_label : str, optional
        dec column name
        Default: dec
    linking_lengths : float, optional
        linking_lengths for FoF matching
        Default: 1.0
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    zp_dict : dict, optional
        magnitude zeropoints dictionary
        Default: {'u': 12.65, 'g': 14.69, 'r': 14.56, 'i': 14.38, 'z': 13.99, 'y':...}
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    match_size : bool, optional
        consider object size for finding blends
        Default: False
    match_shape : bool, optional
        consider object shape for finding blends
        Default: False
    obj_size : str, optional
        object size column name
        Default: obj_size
    a : str, optional
        semi major axis column name
        Default: semi_major
    b : str, optional
        semi minor axis column name
        Default: semi_minor
    theta : str, optional
        orientation angle column name
        Default: orientation

    Returns
    -------
    dict[str, PqHandle]
        A handle giving access to a table with degraded sample
    """
