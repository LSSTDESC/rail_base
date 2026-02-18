from typing import Any

def pz_flow_estimator(**kwargs) -> Any:
    """
    CatEstimator which uses PZFlow

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
    rail.estimation.algos.pzflow_nf.PZFlowEstimator.estimate

    Parameters
    ----------
    input_data : TableLike, required
        A dictionary of all input data
    model : FlowHandle, required
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
    flow_seed : int, optional
        seed for flow
        Default: 0
    ref_column_name : str, optional
        name for reference column
        Default: mag_i_lsst
    column_names : list, optional
        column names to be used in flow
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    mag_limits : dict, optional
        1 sigma mag limits
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    include_mag_errors : bool, optional
        Boolean flag on whether to marginalizeover mag errors (NOTE: much slower on
        CPU!)
        Default: False
    error_names_dict : dict, optional
        dictionary to rename error columns
        Default: {'mag_err_u_lsst': 'mag_u_lsst_err', 'mag_err_g_lsst':...}
    n_error_samples : int, optional
        umber of error samples in marginalization
        Default: 1000
    redshift_column_name : str, optional
        name of redshift column
        Default: redshift

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def pz_flow_informer(**kwargs) -> Any:
    """
    Subclass to train a pzflow-based estimator

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
    rail.estimation.algos.pzflow_nf.PZFlowInformer.inform

    Parameters
    ----------
    training_data : TableLike, required
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    zmin : float, optional
        min z
        Default: 0.0
    zmax : float, optional
        max_z
        Default: 3.0
    nzbins : int, optional
        num z bins
        Default: 301
    flow_seed : int, optional
        seed for flow
        Default: 0
    ref_column_name : str, optional
        name for reference column
        Default: mag_i_lsst
    column_names : list, optional
        column names to be used in flow
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    mag_limits : dict, optional
        1 sigma mag limits
        Default: {'mag_u_lsst': 27.79, 'mag_g_lsst': 29.04, 'mag_r_lsst': 29.06,...}
    include_mag_errors : bool, optional
        Boolean flag on whether to marginalizeover mag errors (NOTE: much slower on
        CPU!)
        Default: False
    error_names_dict : dict, optional
        dictionary to rename error columns
        Default: {'mag_err_u_lsst': 'mag_u_lsst_err', 'mag_err_g_lsst':...}
    n_error_samples : int, optional
        umber of error samples in marginalization
        Default: 1000
    soft_sharpness : int, optional
        sharpening paremeter for SoftPlus
        Default: 10
    soft_idx_col : int, optional
        index column for SoftPlus
        Default: 0
    redshift_column_name : str, optional
        name of redshift column
        Default: redshift
    num_training_epochs : int, optional
        number flow training epochs
        Default: 50

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
