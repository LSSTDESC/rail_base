from typing import Any

def nz_dir_informer(**kwargs) -> Any:
    """
    Quick implementation of an NZ Estimator that creates weights for each input
    object using sklearn's NearestNeighbors.  Very basic, we can probably create a more
    sophisticated SOM-based DIR method in the future.  This inform stage just creates a
    nearneigh model of the spec-z data and some distances to N-th neighbor that will be
    used in the estimate stage.

    This will create `model` a dictionary of the nearest neighbor model and params used
    by estimate

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
    rail.estimation.algos.nz_dir.NZDirInformer.inform

    Parameters
    ----------
    input : TableLike
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    usecols : list, optional
        columns from sz_data for Neighbor calculation
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    n_neigh : int, optional
        number of neighbors to use
        Default: 10
    kalgo : str, optional
        Neighbor algorithm to use
        Default: kd_tree
    kmetric : str, optional
        Knn metric to use
        Default: euclidean
    szname : str, optional
        name of specz column in sz_data
        Default: redshift
    szweightcol : str, optional
        name of sz weight column
        Default:
    distance_delta : float, optional
        padding for distance calculation
        Default: 1e-06

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """

def nz_dir_summarizer(**kwargs) -> Any:
    """
    Quick implementation of a summarizer that creates weights for each input object
    using sklearn's NearestNeighbors.  Very basic, we can probably create a more
    sophisticated SOM-based DIR method in the future

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
    rail.estimation.algos.nz_dir.NZDirSummarizer.estimate

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
        The minimum redshift of the z grid
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid
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
    seed : int, optional
        random seed
        Default: 87
    usecols : list, optional
        columns from sz_date for Neighbor calculation
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    leafsize : int, optional
        leaf size for testdata KDTree
        Default: 40
    phot_weightcol : str, optional
        name of photometry weight, if present
        Default:
    nsamples : int, optional
        number of bootstrap samples to generate
        Default: 20

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """
