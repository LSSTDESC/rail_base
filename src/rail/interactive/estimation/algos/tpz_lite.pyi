from typing import Any

def tpz_lite_estimator(**kwargs) -> Any:
    """
    CatEstimator subclass for regression mode of TPZ
    Requires the trained model with decision trees that are computed by
    TPZliteInformer, and data that has all of the same columns and
    column names as used by that stage!

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
    rail.estimation.algos.tpz_lite.TPZliteEstimator.estimate

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
    err_dict : dict, optional
        dictionary that contains the columns that will be used topredict as the keys and
        the errors associated with that column as the values.If a column does not havea
        an associated error its value shoule be `None`
        Default: {'mag_u_lsst': 'mag_err_u_lsst', 'mag_g_lsst': 'mag_err_g_lsst',...}

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def tpz_lite_informer(**kwargs) -> Any:
    """
    Inform stage for TPZliteEstimator, this stage uses training
    data to train up a set of decision trees that are then stored
    as a pickled model file for use by the Estimator stage.

    n_trees controls how many bootstrap realizations are created from a
    single catalog realization to train one tree.
    nransom controls how many catalog realizations are created. Each
    random catalog consists of adding Gaussian scatter to each attribute
    based on its associated error column.  If the error column `eind` is
    -1 then a small error of 0.00005 is hardcoded into TPZ. The key
    attribute is not included in this random catalog creation.

    So, a total of n_random*n_trees trees are trained and stored in the
    final model i.e. if n_random=3 and n_trees=5 then 15 total trees
    are trained and stored.

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
    rail.estimation.algos.tpz_lite.TPZliteInformer.inform

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
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    seed : int, optional
        random seed
        Default: 8758
    err_dict : dict, optional
        dictionary that contains the columns that will be used topredict as the keys and
        the errors associated with that column as the values.If a column does not havea
        an associated error its value shoule be `None`
        Default: {'mag_u_lsst': 'mag_err_u_lsst', 'mag_g_lsst': 'mag_err_g_lsst',...}
    n_random : int, optional
        number of random bootstrap samples of training data to create
        Default: 8
    n_trees : int, optional
        number of trees to create
        Default: 5
    min_leaf : int, optional
        minimum number in terminal leaf
        Default: 5
    n_att : int, optional
        number of attributes to split for TPZ
        Default: 3
    sigmafactor : float, optional
        Gaussian smoothing with kernel Sigma1*Resolution
        Default: 3.0
    rmsfactor : float, optional
        RMS for zconf calculation
        Default: 0.02
    tree_strategy : str, optional
        which decision tree function to use when constructing the forest, valid choices
        are 'native' or 'sklearn'.  If 'native', use the trees written for TPZ,
        if 'sklearn' then use sklearn's DecisionTreeRegressor
        Default: sklearn

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
