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
    lephare_config_from_model : bool, optional
        A parameter
        Default: True
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
    lephare.ADAPT_BAND : str, optional
        A parameter
        Default: 5
    lephare.ADAPT_CONTEXT : str, optional
        A parameter
        Default: -1
    lephare.ADAPT_LIM : str, optional
        A parameter
        Default: 1.5,23.0
    lephare.ADAPT_MODBIN : str, optional
        A parameter
        Default: 1,1000
    lephare.ADAPT_ZBIN : str, optional
        A parameter
        Default: 0.01,6
    lephare.ADDITIONAL_MAG : str, optional
        A parameter
        Default: none
    lephare.ADD_DUSTEM : str, optional
        A parameter
        Default: NO
    lephare.ADD_EMLINES : str, optional
        A parameter
        Default: 0,10000
    lephare.AGE_RANGE : str, optional
        A parameter
        Default: 0.,15.e9
    lephare.AUTO_ADAPT : str, optional
        A parameter
        Default: NO
    lephare.CAT_FMT : str, optional
        A parameter
        Default: MEME
    lephare.CAT_IN : str, optional
        A parameter
        Default: undefined
    lephare.CAT_LINES : str, optional
        A parameter
        Default: 0,1000000000
    lephare.CAT_MAG : str, optional
        A parameter
        Default: AB
    lephare.CAT_OUT : str, optional
        A parameter
        Default: zphot.out
    lephare.CAT_TYPE : str, optional
        A parameter
        Default: LONG
    lephare.CHI2_OUT : str, optional
        A parameter
        Default: NO
    lephare.COSMOLOGY : str, optional
        A parameter
        Default: 70,0.3,0.7
    lephare.DZ_WIN : str, optional
        A parameter
        Default: 1.0
    lephare.EBV_RANGE : str, optional
        A parameter
        Default: 0,9
    lephare.EB_V : str, optional
        A parameter
        Default: 0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5
    lephare.EM_DISPERSION : str, optional
        A parameter
        Default: 0.5,0.75,1.,1.5,2.
    lephare.EM_LINES : str, optional
        A parameter
        Default: EMP_UV
    lephare.ERR_FACTOR : str, optional
        A parameter
        Default: 1.5
    lephare.ERR_SCALE : str, optional
        A parameter
        Default: 0.02,0.02,0.02,0.02,0.02,0.02
    lephare.EXTERNALZ_FILE : str, optional
        A parameter
        Default: NONE
    lephare.EXTINC_LAW : str, optional
        A parameter
        Default:
        SMC_prevot.dat,SB_calzetti.dat,SB_calzetti_bump1.dat,SB_calzetti_bump2.dat
    lephare.FILTER_CALIB : str, optional
        A parameter
        Default: 0,0,0,0,0,0
    lephare.FILTER_FILE : str, optional
        A parameter
        Default: filter_lsst
    lephare.FILTER_LIST : str, optional
        A parameter
        Default: lsst/total_u.pb,lsst/total_g.pb,lsst/total_r.pb,lsst/total_i.pb,lsst/to
        tal_z.pb,lsst/total_y.pb
    lephare.FILTER_REP : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/filt
    lephare.FIR_CONT : str, optional
        A parameter
        Default: -1
    lephare.FIR_FREESCALE : str, optional
        A parameter
        Default: YES
    lephare.FIR_LIB : str, optional
        A parameter
        Default: NONE
    lephare.FIR_LMIN : str, optional
        A parameter
        Default: 7.0
    lephare.FIR_SCALE : str, optional
        A parameter
        Default: -1
    lephare.FIR_SUBSTELLAR : str, optional
        A parameter
        Default: NO
    lephare.FORB_CONTEXT : str, optional
        A parameter
        Default: -1
    lephare.GAL_FSCALE : str, optional
        A parameter
        Default: 1.
    lephare.GAL_LIB : str, optional
        A parameter
        Default: LSST_GAL_BIN
    lephare.GAL_LIB_IN : str, optional
        A parameter
        Default: LSST_GAL_BIN
    lephare.GAL_LIB_OUT : str, optional
        A parameter
        Default: LSST_GAL_MAG
    lephare.GAL_SED : str, optional
        A parameter
        Default:
        /Users/echarles/Library/Caches/lephare/data/sed/GAL/COSMOS_SED/COSMOS_MOD.list
    lephare.GLB_CONTEXT : str, optional
        A parameter
        Default: 63
    lephare.INP_TYPE : str, optional
        A parameter
        Default: M
    lephare.LIB_ASCII : str, optional
        A parameter
        Default: NO
    lephare.LIMITS_MAPP_CUT : str, optional
        A parameter
        Default: 90
    lephare.LIMITS_MAPP_REF : str, optional
        A parameter
        Default: 1
    lephare.LIMITS_MAPP_SEL : str, optional
        A parameter
        Default: 1
    lephare.LIMITS_ZBIN : str, optional
        A parameter
        Default: 0,99
    lephare.MABS_CONTEXT : str, optional
        A parameter
        Default: 63
    lephare.MABS_FILT : str, optional
        A parameter
        Default: 1,2,3,4
    lephare.MABS_METHOD : str, optional
        A parameter
        Default: 1
    lephare.MABS_REF : str, optional
        A parameter
        Default: 1
    lephare.MABS_ZBIN : str, optional
        A parameter
        Default: 0,0.5,2,4,6
    lephare.MAGTYPE : str, optional
        A parameter
        Default: AB
    lephare.MAG_ABS : str, optional
        A parameter
        Default: -24,-5
    lephare.MAG_ABS_QSO : str, optional
        A parameter
        Default: -30,-10
    lephare.MAG_REF : str, optional
        A parameter
        Default: 3
    lephare.MIN_THRES : str, optional
        A parameter
        Default: 0.02
    lephare.MOD_EXTINC : str, optional
        A parameter
        Default: 0,0
    lephare.PARA_OUT : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/examples/output.para
    lephare.PDZ_OUT : str, optional
        A parameter
        Default: test
    lephare.PDZ_TYPE : str, optional
        A parameter
        Default: BAY_ZG
    lephare.QSO_FSCALE : str, optional
        A parameter
        Default: 1.
    lephare.QSO_LIB : str, optional
        A parameter
        Default: LSST_QSO_BIN
    lephare.QSO_LIB_IN : str, optional
        A parameter
        Default: LSST_QSO_BIN
    lephare.QSO_LIB_OUT : str, optional
        A parameter
        Default: LSST_QSO_MAG
    lephare.QSO_SED : str, optional
        A parameter
        Default:
        /Users/echarles/Library/Caches/lephare/data/sed/QSO/SALVATO09/AGN_MOD.list
    lephare.RF_COLORS : str, optional
        A parameter
        Default: 32,4,4,13
    lephare.RM_DISCREPANT_BD : str, optional
        A parameter
        Default: 500
    lephare.SPEC_OUT : str, optional
        A parameter
        Default: NO
    lephare.STAR_FSCALE : str, optional
        A parameter
        Default: 3.432E-09
    lephare.STAR_LIB : str, optional
        A parameter
        Default: LSST_STAR_BIN
    lephare.STAR_LIB_IN : str, optional
        A parameter
        Default: LSST_STAR_BIN
    lephare.STAR_LIB_OUT : str, optional
        A parameter
        Default: LSST_STAR_MAG
    lephare.STAR_SED : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/sed/STAR/STAR_MOD_ALL.list
    lephare.TRANS_TYPE : str, optional
        A parameter
        Default: 1
    lephare.VERBOSE : str, optional
        A parameter
        Default: NO
    lephare.ZFIX : str, optional
        A parameter
        Default: NO
    lephare.ZGRID_TYPE : str, optional
        A parameter
        Default: 0
    lephare.ZPHOTLIB : str, optional
        A parameter
        Default: LSST_STAR_MAG,LSST_GAL_MAG,LSST_QSO_MAG
    lephare.Z_INTERP : str, optional
        A parameter
        Default: YES
    lephare.Z_METHOD : str, optional
        A parameter
        Default: BEST
    lephare.Z_RANGE : str, optional
        A parameter
        Default: 0.,99.99
    lephare.Z_STEP : str, optional
        A parameter
        Default: 0.01,0.,7.

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
    lephare.ADAPT_BAND : str, optional
        A parameter
        Default: 5
    lephare.ADAPT_CONTEXT : str, optional
        A parameter
        Default: -1
    lephare.ADAPT_LIM : str, optional
        A parameter
        Default: 1.5,23.0
    lephare.ADAPT_MODBIN : str, optional
        A parameter
        Default: 1,1000
    lephare.ADAPT_ZBIN : str, optional
        A parameter
        Default: 0.01,6
    lephare.ADDITIONAL_MAG : str, optional
        A parameter
        Default: none
    lephare.ADD_DUSTEM : str, optional
        A parameter
        Default: NO
    lephare.ADD_EMLINES : str, optional
        A parameter
        Default: 0,10000
    lephare.AGE_RANGE : str, optional
        A parameter
        Default: 0.,15.e9
    lephare.AUTO_ADAPT : str, optional
        A parameter
        Default: NO
    lephare.CAT_FMT : str, optional
        A parameter
        Default: MEME
    lephare.CAT_IN : str, optional
        A parameter
        Default: undefined
    lephare.CAT_LINES : str, optional
        A parameter
        Default: 0,1000000000
    lephare.CAT_MAG : str, optional
        A parameter
        Default: AB
    lephare.CAT_OUT : str, optional
        A parameter
        Default: zphot.out
    lephare.CAT_TYPE : str, optional
        A parameter
        Default: LONG
    lephare.CHI2_OUT : str, optional
        A parameter
        Default: NO
    lephare.COSMOLOGY : str, optional
        A parameter
        Default: 70,0.3,0.7
    lephare.DZ_WIN : str, optional
        A parameter
        Default: 1.0
    lephare.EBV_RANGE : str, optional
        A parameter
        Default: 0,9
    lephare.EB_V : str, optional
        A parameter
        Default: 0.,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5
    lephare.EM_DISPERSION : str, optional
        A parameter
        Default: 0.5,0.75,1.,1.5,2.
    lephare.EM_LINES : str, optional
        A parameter
        Default: EMP_UV
    lephare.ERR_FACTOR : str, optional
        A parameter
        Default: 1.5
    lephare.ERR_SCALE : str, optional
        A parameter
        Default: 0.02,0.02,0.02,0.02,0.02,0.02
    lephare.EXTERNALZ_FILE : str, optional
        A parameter
        Default: NONE
    lephare.EXTINC_LAW : str, optional
        A parameter
        Default:
        SMC_prevot.dat,SB_calzetti.dat,SB_calzetti_bump1.dat,SB_calzetti_bump2.dat
    lephare.FILTER_CALIB : str, optional
        A parameter
        Default: 0,0,0,0,0,0
    lephare.FILTER_FILE : str, optional
        A parameter
        Default: filter_lsst
    lephare.FILTER_LIST : str, optional
        A parameter
        Default: lsst/total_u.pb,lsst/total_g.pb,lsst/total_r.pb,lsst/total_i.pb,lsst/to
        tal_z.pb,lsst/total_y.pb
    lephare.FILTER_REP : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/filt
    lephare.FIR_CONT : str, optional
        A parameter
        Default: -1
    lephare.FIR_FREESCALE : str, optional
        A parameter
        Default: YES
    lephare.FIR_LIB : str, optional
        A parameter
        Default: NONE
    lephare.FIR_LMIN : str, optional
        A parameter
        Default: 7.0
    lephare.FIR_SCALE : str, optional
        A parameter
        Default: -1
    lephare.FIR_SUBSTELLAR : str, optional
        A parameter
        Default: NO
    lephare.FORB_CONTEXT : str, optional
        A parameter
        Default: -1
    lephare.GAL_FSCALE : str, optional
        A parameter
        Default: 1.
    lephare.GAL_LIB : str, optional
        A parameter
        Default: LSST_GAL_BIN
    lephare.GAL_LIB_IN : str, optional
        A parameter
        Default: LSST_GAL_BIN
    lephare.GAL_LIB_OUT : str, optional
        A parameter
        Default: LSST_GAL_MAG
    lephare.GAL_SED : str, optional
        A parameter
        Default:
        /Users/echarles/Library/Caches/lephare/data/sed/GAL/COSMOS_SED/COSMOS_MOD.list
    lephare.GLB_CONTEXT : str, optional
        A parameter
        Default: 63
    lephare.INP_TYPE : str, optional
        A parameter
        Default: M
    lephare.LIB_ASCII : str, optional
        A parameter
        Default: NO
    lephare.LIMITS_MAPP_CUT : str, optional
        A parameter
        Default: 90
    lephare.LIMITS_MAPP_REF : str, optional
        A parameter
        Default: 1
    lephare.LIMITS_MAPP_SEL : str, optional
        A parameter
        Default: 1
    lephare.LIMITS_ZBIN : str, optional
        A parameter
        Default: 0,99
    lephare.MABS_CONTEXT : str, optional
        A parameter
        Default: 63
    lephare.MABS_FILT : str, optional
        A parameter
        Default: 1,2,3,4
    lephare.MABS_METHOD : str, optional
        A parameter
        Default: 1
    lephare.MABS_REF : str, optional
        A parameter
        Default: 1
    lephare.MABS_ZBIN : str, optional
        A parameter
        Default: 0,0.5,2,4,6
    lephare.MAGTYPE : str, optional
        A parameter
        Default: AB
    lephare.MAG_ABS : str, optional
        A parameter
        Default: -24,-5
    lephare.MAG_ABS_QSO : str, optional
        A parameter
        Default: -30,-10
    lephare.MAG_REF : str, optional
        A parameter
        Default: 3
    lephare.MIN_THRES : str, optional
        A parameter
        Default: 0.02
    lephare.MOD_EXTINC : str, optional
        A parameter
        Default: 0,0
    lephare.PARA_OUT : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/examples/output.para
    lephare.PDZ_OUT : str, optional
        A parameter
        Default: test
    lephare.PDZ_TYPE : str, optional
        A parameter
        Default: BAY_ZG
    lephare.QSO_FSCALE : str, optional
        A parameter
        Default: 1.
    lephare.QSO_LIB : str, optional
        A parameter
        Default: LSST_QSO_BIN
    lephare.QSO_LIB_IN : str, optional
        A parameter
        Default: LSST_QSO_BIN
    lephare.QSO_LIB_OUT : str, optional
        A parameter
        Default: LSST_QSO_MAG
    lephare.QSO_SED : str, optional
        A parameter
        Default:
        /Users/echarles/Library/Caches/lephare/data/sed/QSO/SALVATO09/AGN_MOD.list
    lephare.RF_COLORS : str, optional
        A parameter
        Default: 32,4,4,13
    lephare.RM_DISCREPANT_BD : str, optional
        A parameter
        Default: 500
    lephare.SPEC_OUT : str, optional
        A parameter
        Default: NO
    lephare.STAR_FSCALE : str, optional
        A parameter
        Default: 3.432E-09
    lephare.STAR_LIB : str, optional
        A parameter
        Default: LSST_STAR_BIN
    lephare.STAR_LIB_IN : str, optional
        A parameter
        Default: LSST_STAR_BIN
    lephare.STAR_LIB_OUT : str, optional
        A parameter
        Default: LSST_STAR_MAG
    lephare.STAR_SED : str, optional
        A parameter
        Default: /Users/echarles/Library/Caches/lephare/data/sed/STAR/STAR_MOD_ALL.list
    lephare.TRANS_TYPE : str, optional
        A parameter
        Default: 1
    lephare.VERBOSE : str, optional
        A parameter
        Default: NO
    lephare.ZFIX : str, optional
        A parameter
        Default: NO
    lephare.ZGRID_TYPE : str, optional
        A parameter
        Default: 0
    lephare.ZPHOTLIB : str, optional
        A parameter
        Default: LSST_STAR_MAG,LSST_GAL_MAG,LSST_QSO_MAG
    lephare.Z_INTERP : str, optional
        A parameter
        Default: YES
    lephare.Z_METHOD : str, optional
        A parameter
        Default: BEST
    lephare.Z_RANGE : str, optional
        A parameter
        Default: 0.,99.99
    lephare.Z_STEP : str, optional
        A parameter
        Default: 0.01,0.,7.
    star.LIB_ASCII : str, optional
        A parameter
        Default: YES
    gal.LIB_ASCII : str, optional
        A parameter
        Default: YES
    gal.MOD_EXTINC : str, optional
        A parameter
        Default: 18,26,26,33,26,33,26,33
    gal.EXTINC_LAW : str, optional
        A parameter
        Default:
        SMC_prevot.dat,SB_calzetti.dat,SB_calzetti_bump1.dat,SB_calzetti_bump2.dat
    gal.EM_LINES : str, optional
        A parameter
        Default: EMP_UV
    gal.EM_DISPERSION : str, optional
        A parameter
        Default: 0.5,0.75,1.,1.5,2.
    qso.LIB_ASCII : str, optional
        A parameter
        Default: YES
    qso.MOD_EXTINC : str, optional
        A parameter
        Default: 0,1000
    qso.EB_V : str, optional
        A parameter
        Default: 0.,0.1,0.2,0.3
    qso.EXTINC_LAW : str, optional
        A parameter
        Default: SB_calzetti.dat

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
