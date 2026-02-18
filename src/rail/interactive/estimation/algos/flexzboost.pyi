from typing import Any

def flex_z_boost_estimator(**kwargs) -> Any:
    """
    FlexZBoost-based CatEstimator

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
    rail.estimation.algos.flexzboost.FlexZBoostEstimator.estimate

    Parameters
    ----------
    input_data : TableLike, required
        A dictionary of all input data
    model : numpy.ndarray, required
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
    qp_representation : str, optional
        qp generator to use. [interp|flexzboost]
        Default: interp
    include_mag_err : bool, optional
        Include magnitude error in the training and estimationprocess
        Default: False

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def flex_z_boost_informer(**kwargs) -> Any:
    """
    Train a FlexZBoost CatInformer

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
    rail.estimation.algos.flexzboost.FlexZBoostInformer.inform

    Parameters
    ----------
    training_data : TableLike, required
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
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    retrain_full : bool, optional
        if True, re-run the fit with the full training set, including data set aside for
        bump/sharpen validation.  If False, only use the subset defined via trainfrac
        fraction
        Default: True
    trainfrac : float, optional
        fraction of training data to use for training (rest used for bump thresh and
        sharpening determination)
        Default: 0.75
    seed : int, optional
        Random number seed
        Default: 1138
    bumpmin : float, optional
        minimum value in grid of thresholds checked to optimize removal of spurious
        small bumps
        Default: 0.02
    bumpmax : float, optional
        max value in grid checked for removal of small bumps
        Default: 0.35
    nbump : int, optional
        number of grid points in bumpthresh grid search
        Default: 20
    sharpmin : float, optional
        min value in grid checked in optimal sharpening parameter fit
        Default: 0.7
    sharpmax : float, optional
        max value in grid checked in optimal sharpening parameter fit
        Default: 2.1
    nsharp : int, optional
        number of search points in sharpening fit
        Default: 15
    max_basis : int, optional
        maximum number of basis funcitons to use in density estimate
        Default: 35
    basis_system : str, optional
        type of basis sytem to use with flexcode
        Default: cosine
    regression_params : dict, optional
        dictionary of options passed to flexcode, includes max_depth (int), and
        objective, which should be set  to reg:squarederror
        Default: {'max_depth': 8, 'objective': 'reg:squarederror'}
    include_mag_err : bool, optional
        Include magnitude error in the training and estimationprocess
        Default: False

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
