from typing import Any

def gpz_estimator(**kwargs) -> Any:
    """
    Estimate stage for GPz_v1

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
    rail.estimation.algos.gpz.GPzEstimator.estimate

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
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    log_errors : bool, optional
        if true, take log of magnitude errors
        Default: True
    replace_error_vals : list, optional
        list of values to replace negative and nan mag err values
        Default: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def gpz_informer(**kwargs) -> Any:
    """
    Inform stage for GPz_v1

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
    rail.estimation.algos.gpz.GPzInformer.inform

    Parameters
    ----------
    input : TableLike
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    trainfrac : float, optional
        fraction of training data used to make tree, rest used to set best sigma
        Default: 0.75
    seed : int, optional
        random seed
        Default: 87
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    gpz_method : str, optional
        method to be used in GPz, options are 'GL', 'VL', 'GD', 'VD', 'GC', and 'VC'
        Default: VC
    n_basis : int, optional
        number of basis functions used
        Default: 50
    learn_jointly : bool, optional
        if True, jointly learns prior linear mean function
        Default: True
    hetero_noise : bool, optional
        if True, learns heteroscedastic noise process, set False for point est.
        Default: True
    csl_method : str, optional
        cost sensitive learning type, 'balanced', 'normalized', or 'normal'
        Default: normal
    csl_binwidth : float, optional
        width of bin for 'balanced' cost sensitive learning
        Default: 0.1
    pca_decorrelate : bool, optional
        if True, decorrelate data using PCA as preprocessing stage
        Default: True
    max_iter : int, optional
        max number of iterations
        Default: 200
    max_attempt : int, optional
        max iterations if no progress on validation
        Default: 100
    log_errors : bool, optional
        if true, take log of magnitude errors
        Default: True
    replace_error_vals : list, optional
        list of values to replace negative and nan mag err values
        Default: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
