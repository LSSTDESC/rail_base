from typing import Any

def cmnn_estimator(**kwargs) -> Any:
    """
    Color Matched Nearest Neighbor Estimator
    Note that there are several modifications from the original CMNN, mainly that
    the original estimator dropped non-detections from the Mahalnobis distance
    calculation. However, there is information in a non-detection, so instead here
    I've replaced the non-detections with 1 sigma limit and a magnitude
    uncertainty of 1.0 and fixed the degrees of freedom to be the number of
    magnitude bands minus one.

    Current implementation returns a single Gaussian for each galaxy with a width
    determined by the std deviation of all galaxies within the range set by the
    ppf value.

    There are three options for how to choose the central value of the Gaussian
    and that option is set using the `selection_mode` config parameter (integer):
    option 0: randomly choose one of the neighbors within the PPF cutoff
    option 1: choose the value with the smallest Mahalnobis distance
    option 2: random choice as in option 0, but weighted by distance

    If a test galaxy does not have enough training galaxies it is
    assigned a redshift `bad_redshift_val` and a width `bad_redshift_err`, both
    of which are config parameters that can be set by the user.  Note that this
    should only happen if the number of training galaxies is smaller than
    min_n, which is unlikely, but is included here for completeness.

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
    rail.estimation.algos.cmnn.CMNNEstimator.estimate

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
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    seed : int, optional
        random seed used in selection mode
        Default: 66
    ppf_value : float, optional
        PPF value used in Mahalanobis distance
        Default: 0.68
    selection_mode : int, optional
        select which mode to choose the redshift estimate:0: randomly choose, 1: nearest
        neigh, 2: weighted random
        Default: 1
    min_n : int, optional
        minimum number of training galaxies to use
        Default: 25
    min_thresh : float, optional
        minimum threshold cutoff
        Default: 0.0001
    min_dist : float, optional
        minimum Mahalanobis distance
        Default: 0.0001
    bad_redshift_val : float, optional
        redshift to assign bad redshifts
        Default: 99.0
    bad_redshift_err : float, optional
        Gauss error width to assign to bad redshifts
        Default: 10.0

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def cmnn_informer(**kwargs) -> Any:
    """
    compute colors and color errors for CMNN training set and
    store in a model file that will be used by the CMNNEstimator stage

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
    rail.estimation.algos.cmnn.CMNNInformer.inform

    Parameters
    ----------
    input : TableLike
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    bands : list, optional
        Names of columns for magnitude by filter band
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    mag_limits : dict, optional
        Limiting magnitudes by filter
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    nondetect_replace : bool, optional
        set to True to replace non-detects, False to ignore in distance calculation
        Default: False

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
