from typing import Any

def lephare_estimator(**kwargs) -> Any:
    """
    LePhare-base CatEstimator

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
    rail.estimation.algos.lephare.LephareEstimator.estimate

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
    ref_band : str, optional
        band to use in addition to colors
        Default: mag_i_lsst
    err_bands : list, optional
        Names of columns for magnitude errors by filter band
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    lephare_config : dict, optional
        The lephare config keymap. If unset we load it from the model.
        Default: {}
    use_inform_offsets : bool, optional
        Use the zero point offsets computed in the inform stage.
        Default: True
    posterior_output : int, optional
        Which posterior distribution to output.MASS: 0SFR: 1SSFR: 2LDUST: 3LIR: 4AGE:
        5COL1: 6COL2: 7MREF: 8MIN_ZG: 9MIN_ZQ: 10BAY_ZG: 11BAY_ZQ: 12
        Default: 11
    output_keys : list, optional
        The output keys to add to ancil. These must be in the output para file. By
        default we include the best galaxy and QSO redshift and best star alongside
        their respective chi squared.
        Default: ['Z_BEST', 'CHI_BEST', 'ZQ_BEST', 'CHI_QSO', 'MOD_STAR', 'CHI_STAR']
    run_dir : str, optional
        Override for the LEPHAREWORK directory. If None we load it from the model which
        is set during the inform stage. This is to facilitate manually moving
        intermediate files.
        Default: None

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def lephare_informer(**kwargs) -> Any:
    """
    Inform stage for LephareEstimator

    This class will set templates and filters required for photoz estimation.

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
    rail.estimation.algos.lephare.LephareInformer.inform

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
    lephare_config : dict, optional
        The lephare config keymap.
        Default: {'ADAPT_BAND': '5', 'ADAPT_CONTEXT': '-1', 'ADAPT_LIM':...}
    star_config : dict, optional
        Star config overrides.
        Default: {'LIB_ASCII': 'YES'}
    gal_config : dict, optional
        Galaxy config overrides.
        Default: {'LIB_ASCII': 'YES', 'MOD_EXTINC': '18,26,26,33,26,33,26,33',...}
    qso_config : dict, optional
        QSO config overrides.
        Default: {'LIB_ASCII': 'YES', 'MOD_EXTINC': '0,1000', 'EB_V':...}

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
