from typing import Any

def bpz_lite_estimator(**kwargs) -> Any:
    """
    CatEstimator subclass to implement basic marginalized PDF for BPZ
    In addition to the marginalized redshift PDF, we also compute several
    ancillary quantities that will be stored in the ensemble ancil data:
    zmode: mode of the PDF
    amean: mean of the PDF
    tb: integer specifying the best-fit SED *at the redshift mode*
    todds: fraction of marginalized posterior prob. of best template,
    so lower numbers mean other templates could be better fits, likely
    at other redshifts

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
    rail.estimation.algos.bpz_lite.BPZliteEstimator.estimate

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
    dz : float, optional
        delta z in grid
        Default: 0.01
    unobserved_val : float, optional
        value to be replaced with zero flux and given large errors for non-observed
        filters
        Default: -99.0
    data_path : str, optional
        data_path (str): file path to the SED, FILTER, and AB directories.  If left to
        default `None` it will use the install directory for rail +
        ../examples_data/estimation_data/data
        Default: None
    filter_list : list, optional
        list of filter files names (with no '.sed' suffix). Filters must bein FILTER
        dir.  MUST BE IN SAME ORDER as 'bands'
        Default: ['DC2LSST_u', 'DC2LSST_g', 'DC2LSST_r', 'DC2LSST_i', 'DC2LSST_z',...]
    spectra_file : str, optional
        name of the file specifying the list of SEDs to use
        Default: CWWSB4.list
    madau_flag : str, optional
        set to 'yes' or 'no' to set whether to include intergalactic Madau reddening
        when constructing model fluxes
        Default: no
    no_prior : bool, optional
        set to True if you want to run with no prior
        Default: False
    p_min : float, optional
        BPZ sets all values of the PDF that are below p_min*peak_value to 0.0, p_min
        controls that fractional cutoff
        Default: 0.005
    gauss_kernel : float, optional
        gauss_kernel (float): BPZ convolves the PDF with a kernel if this is set to a
        non-zero number
        Default: 0.0
    zp_errors : list, optional
        BPZ adds these values in quadrature to the photometric errors
        Default: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    mag_err_min : float, optional
        a minimum floor for the magnitude errors to prevent a large chi^2 for very very
        bright objects
        Default: 0.005

    Returns
    -------
    qp.core.ensemble.Ensemble
        Handle providing access to QP ensemble with output data
    """

def bpz_lite_informer(**kwargs) -> Any:
    """
    Inform stage for BPZliteEstimator, this stage *assumes* that you have a set of
    SED templates and that the training data has already been assigned a
    'best fit broad type' (that is, something like ellliptical, spiral,
    irregular, or starburst, similar to how the six SEDs in the CWW/SB set
    of Benitez (2000) are assigned 3 broad types).  This informer will then
    fit parameters for the evolving type fraction as a function of apparent
    magnitude in a reference band, P(T|m), as well as the redshift prior
    of finding a galaxy of the broad type at a particular redshift, p(z|m, T)
    where z is redshift, m is apparent magnitude in the reference band, and T
    is the 'broad type'.  We will use the same forms for these functions as
    parameterized in Benitez (2000).  For p(T|m) we have
    p(T|m) = exp(-kt(m-m0))
    where m0 is a constant and we fit for values of kt
    For p(z|T,m) we have

    ```
    P(z|T,m) = f_x*z0_x^a *exp(-(z/zm_x)^a)
    where zm_x = z0_x*(km_x-m0)
    ```

    where f_x is the type fraction from p(T|m), and we fit for values of
    z0, km, and a for each type.  These parameters are then fed to the BPZ
    prior for use in the estimation stage.

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
    rail.estimation.algos.bpz_lite.BPZliteInformer.inform

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
    data_path : str, optional
        data_path (str): file path to the SED, FILTER, and AB directories.  If left to
        default `None` it will use the install directory for rail +
        rail/examples_data/estimation_data/data
        Default: None
    spectra_file : str, optional
        name of the file specifying the list of SEDs to use
        Default: CWWSB4.list
    m0 : float, optional
        reference apparent mag, used in prior param
        Default: 20.0
    nt_array : list, optional
        list of integer number of templates per 'broad type', must be in same order as
        the template set, and must sum to the same number as the # of templates in the
        spectra file
        Default: [1, 2, 5]
    mmin : float, optional
        lowest apparent mag in ref band, lower values ignored
        Default: 18.0
    mmax : float, optional
        highest apparent mag in ref band, higher values ignored
        Default: 29.0
    init_kt : float, optional
        initial guess for kt in training
        Default: 0.3
    init_zo : float, optional
        initial guess for z0 in training
        Default: 0.4
    init_alpha : float, optional
        initial guess for alpha in training
        Default: 1.8
    init_km : float, optional
        initial guess for km in training
        Default: 0.1
    type_file : str, optional
        name of file with the broad type fits for the training data
        Default:
    output_hdfn : bool, optional
        if True, just return the default HDFN prior params rather than fitting
        Default: True

    Returns
    -------
    numpy.ndarray
        Handle providing access to trained model
    """
