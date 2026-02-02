from typing import Any

def k_near_neigh_estimator(**kwargs) -> Any:
    """
    KNN-based estimator

    ---

    The main interface method for the photo-z estimation

    This will attach the input data (defined in ``inputs`` as "input") to this
    ``Estimator`` (for introspection and provenance tracking). Then call the
    ``run()``, ``validate()``, and ``finalize()`` methods.

    The run method will call ``_process_chunk()``, which needs to be implemented
    in the subclass, to process input data in batches. See ``RandomGaussEstimator``
    for a simple example.

    Finally, this will return a ``QPHandle`` for access to that output data.

    ---

    This function was generated from the function
    rail.estimation.algos.k_nearneigh.KNearNeighEstimator.estimate

    Parameters
    ----------
    input : TableLike
        A dictionary of all input data
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    id_col : str, optional
        name of the object ID column
        Default: object_id
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    calc_summary_stats : bool, optional
        Compute summary statistics
        Default: False
    calculated_point_estimates : list, optional
        List of strings defining which point estimates to automatically calculate using
        `qp.Ensemble`.Options include, 'mean', 'mode', 'median'.
        Default: []
    recompute_point_estimates : bool, optional
        Force recomputation of point estimates
        Default: False
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def k_near_neigh_informer(**kwargs) -> Any:
    """
    Train a KNN-based estimator

    ---

    The main interface method for Informers

    This will attach the input_data to this `Informer`
    (for introspection and provenance tracking).

    Then it will call the run(), validate() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the model that it creates to this Estimator
    by using `self.add_data('model', model)`.

    Finally, this will return a ModelHandle providing access to the trained model.

    ---

    This function was generated from the function
    rail.estimation.algos.k_nearneigh.KNearNeighInformer.inform

    Parameters
    ----------
    input : TableLike
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    trainfrac : float, optional
        fraction of training data used to make tree, rest used to set best sigma
        Default: 0.75
    seed : int, optional
        Random number seed for NN training
        Default: 0
    sigma_grid_min : float, optional
        minimum value of sigma for grid check
        Default: 0.01
    sigma_grid_max : float, optional
        maximum value of sigma for grid check
        Default: 0.075
    ngrid_sigma : int, optional
        number of grid points in sigma check
        Default: 10
    leaf_size : int, optional
        min leaf size for KDTree
        Default: 15
    nneigh_min : int, optional
        int, min number of near neighbors to use for PDF fit
        Default: 3
    nneigh_max : int, optional
        int, max number of near neighbors to use ofr PDF fit
        Default: 7
    only_colors : bool, optional
        if only_colors True, then do not use ref_band mag, only use colors
        Default: False

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
