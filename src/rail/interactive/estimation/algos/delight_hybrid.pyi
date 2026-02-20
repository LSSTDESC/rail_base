from typing import Any

def delight_estimator(**kwargs) -> Any:
    """
    Run the delight scripts from the LSSTDESC fork of Delight
    Still has the ascii writeout stuff, so intermediate files are
    created that need to be cleaned up in the future

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
    rail.estimation.algos.delight_hybrid.DelightEstimator.estimate

    Parameters
    ----------
    input_data : TableLike, required
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
        num z bins
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
    dlght_redshiftMin : float, optional
        min redshift
        Default: 0.01
    dlght_redshiftMax : float, optional
        max redshift
        Default: 3.01
    dlght_redshiftNumBinsGPpred : int, optional
        num bins
        Default: 301
    dlght_redshiftBinSize : float, optional
        ???
        Default: 0.01
    dlght_redshiftDisBinSize : float, optional
        bad, shouldnt be here
        Default: 0.2
    bands_names : str, optional
        string with list of Filter names
        Default: DC2LSST_u DC2LSST_g DC2LSST_r DC2LSST_i DC2LSST_z DC2LSST_y
    bands_path : str, optional
        string specifying path to filter directory
        Default: rail/examples_data/estimation_data/data/FILTER
    bands_fmt : str, optional
        string giving the file extension of the filters, not including the '.'
        Default: res
    bands_numcoefs : int, optional
        integer specifying number of coefs in approximation of filter
        Default: 15
    bands_verbose : bool, optional
        verbose
        Default: True
    bands_makeplots : bool, optional
        bool for whether to make approx band plots
        Default: False
    bands_debug : bool, optional
        debug flag for filters
        Default: True
    tempdir : str, optional
        temp dir
        Default: rail/examples_data/estimation_data/tmp
    tempdatadir : str, optional
        temp data dir
        Default: rail/examples_data/estimation_data/tmp/delight_data
    sed_path : str, optional
        path to SED dir
        Default: rail/examples_data/estimation_data/data/SED
    sed_name_list : str, optional
        String with list of all SED names, with no file extension
        Default: El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a
        ssp_25Myr_z008 ssp_5Myr_z008
    sed_fmt : str, optional
        file extension of SED files (withough the '.', e.g dat or sed
        Default: sed
    prior_t_list : str, optional
        String of numbers specifying prior type fracs MUST BE SAME LENGTH AS NUMBER OF
        SEDS
        Default: 0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079
    prior_zt_list : str, optional
        string of numbers for redshift prior, MUST BE SAME LENGTH AS NUMBER OF SEDS
        Default: 0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14
    lambda_ref : float, optional
        referebce wavelength
        Default: 4500.0
    train_refbandorder : str, optional
        order of bands used in training
        Default: DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var
        DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift
    train_refband : str, optional
        reference band
        Default: DC2LSST_i
    train_fracfluxerr : float, optional
        frac err to add to flux?
        Default: 0.0001
    train_xvalidate : bool, optional
        perform cross validation flag
        Default: False
    train_xvalbandorder : str, optional
        band order for xval, unused bands indicated with _
        Default: _ _ _ _ DC2LSST_r DC2LSST_r_var _ _ _ _ _ _
    gp_params_file : str, optional
        name of file to store gaussian process params fit by delightLearn
        Default: galaxies_gpparams.txt
    crossval_file : str, optional
        name of file to store crossvalidation parameters from delightLearn
        Default: galaxies-gpCV.txt
    target_refbandorder : str, optional
        order of reference bands for target data
        Default: DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var
        DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift
    target_refband : str, optional
        the reference band for the taret data
        Default: DC2LSST_r
    target_fracfluxerr : float, optional
        extra fractional error to add to target fluxes?
        Default: 0.0001
    delightparamfile : str, optional
        param file name
        Default: parametersTest.cfg
    flag_filter_training : bool, optional
        ?
        Default: True
    snr_cut_training : float, optional
        SNR training cut
        Default: 5
    flag_filter_validation : bool, optional
        ?
        Default: True
    snr_cut_validation : float, optional
        validation SNR cut
        Default: 3
    dlght_inputdata : str, optional
        input data directory for ascii data
        Default: rail/examples_data/estimation_data/tmp/delight_indata
    zPriorSigma : float, optional
        sigma for redshift prior
        Default: 0.2
    ellPriorSigma : float, optional
        prior param
        Default: 0.5
    fluxLuminosityNorm : float, optional
        luminosity norm factor
        Default: 1.0
    alpha_C : float, optional
        prior param
        Default: 1000.0
    V_C : float, optional
        prior param
        Default: 0.1
    alpha_L : float, optional
        prior param
        Default: 100.0
    V_L : float, optional
        prior param
        Default: 0.1
    lineWidthSigma : float, optional
        prior param
        Default: 20

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def delight_informer(**kwargs) -> Any:
    """
    Train the Delight code, outputs are actually saved to files,
    which is fairly non-standard way currently

    ---

    Override the inform method because Delight doesn't have a model to return

    ---

    This function was generated from the function
    rail.estimation.algos.delight_hybrid.DelightInformer.inform

    Parameters
    ----------
    training_data : `dict` or `TableHandle`, required
        dictionary of all input data, or a `TableHandle` providing access to it
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    dlght_redshiftMin : float, optional
        min redshift
        Default: 0.01
    dlght_redshiftMax : float, optional
        max redshift
        Default: 3.01
    dlght_redshiftNumBinsGPpred : int, optional
        num bins
        Default: 301
    nzbins : int, optional
        num z bins
        Default: 301
    dlght_redshiftBinSize : float, optional
        ???
        Default: 0.01
    dlght_redshiftDisBinSize : float, optional
        bad, shouldnt be here
        Default: 0.2
    bands_names : str, optional
        string with list of Filter names
        Default: DC2LSST_u DC2LSST_g DC2LSST_r DC2LSST_i DC2LSST_z DC2LSST_y
    bands_path : str, optional
        string specifying path to filter directory
        Default: rail/examples_data/estimation_data/data/FILTER
    bands_fmt : str, optional
        string giving the file extension of the filters, not including the '.'
        Default: res
    bands_numcoefs : int, optional
        integer specifying number of coefs in approximation of filter
        Default: 15
    bands_verbose : bool, optional
        verbose
        Default: True
    bands_makeplots : bool, optional
        bool for whether to make approx band plots
        Default: False
    bands_debug : bool, optional
        debug flag for filters
        Default: True
    tempdir : str, optional
        temp dir
        Default: rail/examples_data/estimation_data/tmp
    tempdatadir : str, optional
        temp data dir
        Default: rail/examples_data/estimation_data/tmp/delight_data
    sed_path : str, optional
        path to SED dir
        Default: rail/examples_data/estimation_data/data/SED
    sed_name_list : str, optional
        String with list of all SED names, with no file extension
        Default: El_B2004a Sbc_B2004a Scd_B2004a SB3_B2004a SB2_B2004a Im_B2004a
        ssp_25Myr_z008 ssp_5Myr_z008
    sed_fmt : str, optional
        file extension of SED files (withough the '.', e.g dat or sed
        Default: sed
    prior_t_list : str, optional
        String of numbers specifying prior type fracs MUST BE SAME LENGTH AS NUMBER OF
        SEDS
        Default: 0.27 0.26 0.25 0.069 0.021 0.11 0.0061 0.0079
    prior_zt_list : str, optional
        string of numbers for redshift prior, MUST BE SAME LENGTH AS NUMBER OF SEDS
        Default: 0.23 0.39 0.33 0.31 1.1 0.34 1.2 0.14
    lambda_ref : float, optional
        referebce wavelength
        Default: 4500.0
    train_refbandorder : str, optional
        order of bands used in training
        Default: DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var
        DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift
    train_refband : str, optional
        reference band
        Default: DC2LSST_i
    train_fracfluxerr : float, optional
        frac err to add to flux?
        Default: 0.0001
    train_xvalidate : bool, optional
        perform cross validation flag
        Default: False
    train_xvalbandorder : str, optional
        band order for xval, unused bands indicated with _
        Default: _ _ _ _ DC2LSST_r DC2LSST_r_var _ _ _ _ _ _
    gp_params_file : str, optional
        name of file to store gaussian process params fit by delightLearn
        Default: galaxies_gpparams.txt
    crossval_file : str, optional
        name of file to store crossvalidation parameters from delightLearn
        Default: galaxies-gpCV.txt
    target_refbandorder : str, optional
        order of reference bands for target data
        Default: DC2LSST_u DC2LSST_u_var DC2LSST_g DC2LSST_g_var DC2LSST_r DC2LSST_r_var
        DC2LSST_i DC2LSST_i_var DC2LSST_z DC2LSST_z_var DC2LSST_y DC2LSST_y_var redshift
    target_refband : str, optional
        the reference band for the taret data
        Default: DC2LSST_r
    target_fracfluxerr : float, optional
        extra fractional error to add to target fluxes?
        Default: 0.0001
    delightparamfile : str, optional
        param file name
        Default: parametersTest.cfg
    flag_filter_training : bool, optional
        ?
        Default: True
    snr_cut_training : float, optional
        SNR training cut
        Default: 5
    flag_filter_validation : bool, optional
        ?
        Default: True
    snr_cut_validation : float, optional
        validation SNR cut
        Default: 3
    dlght_inputdata : str, optional
        input data directory for ascii data
        Default: rail/examples_data/estimation_data/tmp/delight_indata
    zPriorSigma : float, optional
        sigma for redshift prior
        Default: 0.2
    ellPriorSigma : float, optional
        prior param
        Default: 0.5
    fluxLuminosityNorm : float, optional
        luminosity norm factor
        Default: 1.0
    alpha_C : float, optional
        prior param
        Default: 1000.0
    V_C : float, optional
        prior param
        Default: 0.1
    alpha_L : float, optional
        prior param
        Default: 100.0
    V_L : float, optional
        prior param
        Default: 0.1
    lineWidthSigma : float, optional
        prior param
        Default: 20

    Returns
    -------
    None
    """
