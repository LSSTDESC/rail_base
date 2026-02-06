from typing import Any

def yaw_auto_correlate(**kwargs) -> Any:
    """
    Wrapper stage for `yaw.autocorrelate` to compute a sample's angular
    autocorrelation amplitude.

    Generally used for the reference sample to compute an estimate for its
    galaxy sample as a function of redshift. Data is provided as a single cache
    directory that must have redshifts and randoms with redshift attached.

    ---

    Measure the angular autocorrelation amplitude in bins of redshift.

    ---

    This function was generated from the function
    rail.estimation.algos.cc_yaw.YawAutoCorrelate.correlate

    Parameters
    ----------
    sample : YawCache, required
        Input cache which must have randoms attached and redshifts for both
        data set and randoms.
    rmin : float, required
        Single or sequence of lower scale limits in given 'unit'.
    rmax : float, required
        Single or sequence of upper scale limits in given 'unit'.
    unit : str, optional
        The unit of the lower and upper scale limits.
        Default: kpc
    rweight : float, optional
        Power-law exponent used to weight pairs by their separation.
        Default: None
    resolution : int, optional
        Number of radial logarithmic bin used to approximate the weighting by
        separation.
        Default: None
    zmin : float, optional
        Lowest redshift bin edge to generate (alternatively use 'edges').
        Default: None
    zmax : float, optional
        Highest redshift bin edge to generate (alternatively use 'edges').
        Default: None
    num_bins : int, optional
        Number of redshift bins to generate between 'zmin' and 'zmax'.
        Default: 30
    method : str, optional
        Method used to compute the spacing of bin edges.
        Default: linear
    edges : float, optional
        Use these custom bin edges instead of generating them.
        Default: None
    closed : str, optional
        String indicating the side of the bin intervals that are closed.
        Default: right
    max_workers : int, optional
        configure a custom maximum number of parallel workers to use
        Default: None
    verbose : str, optional
        lowest log level emitted by *yet_another_wizz*
        Default: info

    Returns
    -------
    YawCorrFuncHandle
        A handle for the `yaw.CorrFunc` instance that holds the pair counts.
    """

def yaw_cache_create(**kwargs) -> Any:
    """
    Create a new cache directory to hold a data set and optionally its matching
    random catalog.

    Both input data sets are split into consistent spatial patches that are
    required by *yet_another_wizz* for correlation function covariance
    estimates. Each patch is stored separately for efficient access.

    The cache can be constructed from input files or tabular data in memory.
    Column names for sky coordinates are required, redshifts and per-object
    weights are optional. One out of three patch create methods must be
    specified:

    #. Splitting the data into predefined patches (from ASCII file or an
       existing cache instance, linked as optional stage input).
    #. Splitting the data based on a column with patch indices.
    #. Generating approximately equal size patches using k-means clustering of
       objects positions (preferably randoms if provided).

    **Note:** The cache directory must be deleted manually when it is no longer
    needed. (The reference sample cache may be reused when operating on
    tomographic bins.)

    ---

    Create the new cache directory and split the input data into spatial
    patches.

    ---

    This function was generated from the function
    rail.estimation.algos.cc_yaw.YawCacheCreate.create

    Parameters
    ----------
    data : DataFrame, required
        The data set to split into patches and cache.
    path : str, required
        path to cache directory, must not exist
    rand : DataFrame, required
        The randoms to split into patches and cache, positions used to
        automatically generate patch centers if provided and stage is
        configured with `patch_num`.
    patch_source : YawCache, required
        An existing cache instance that provides the patch centers. Use to
        ensure consistent patch centers when running cross-correlations.
        Takes precedence over the any configuration parameters.
    overwrite : bool, optional
        overwrite the path if it is an existing cache directory
        Default: None
    ra_name : str, optional
        column name of right ascension (in degrees)
        Default: ra
    dec_name : str, optional
        column name of declination (in degrees)
        Default: dec
    weight_name : str, optional
        column name of weight
        Default: None
    redshift_name : str, optional
        column name of redshift
        Default: None
    degrees : bool, optional
        Whether the input coordinates are in degrees or radian.
        Default: True
    patch_file : str, optional
        path to ASCII file that lists patch centers (one per line) as pair of R.A./Dec.
        in radian, separated by a single space or tab
        Default: None
    patch_name : str, optional
        column name of patch index (starting from 0)
        Default: None
    patch_num : int, optional
        number of spatial patches to create using knn on coordinates of randoms
        Default: None
    probe_size : int, optional
        The approximate number of objects to sample from the input file when generating
        patch centers.
        Default: -1
    max_workers : int, optional
        configure a custom maximum number of parallel workers to use
        Default: None
    verbose : str, optional
        lowest log level emitted by *yet_another_wizz*
        Default: info

    Returns
    -------
    YawCacheHandle
        A handle for the newly created cache directory.
    """

def yaw_cross_correlate(**kwargs) -> Any:
    """
    Wrapper stage for `yaw.crosscorrelate` to compute the angular cross-
    correlation amplitude between the reference and the unknown sample.

    Generally used for the reference sample to compute an estimate for its
    galaxy sample as a function of redshift. Data sets are provided as cache
    directories. The reference sample must have redshifts and at least one
    cache must have randoms attached.

    ---

    Measure the angular cross-correlation amplitude in bins of redshift.

    ---

    This function was generated from the function
    rail.estimation.algos.cc_yaw.YawCrossCorrelate.correlate

    Parameters
    ----------
    reference : YawCache, required
        Cache for the reference data, must have redshifts. If no randoms are
        attached, the unknown data cache must provide them.
    unknown : YawCache, required
        Cache for the unknown data. If no randoms are attached, the
        reference data cache must provide them.
    rmin : float, required
        Single or sequence of lower scale limits in given 'unit'.
    rmax : float, required
        Single or sequence of upper scale limits in given 'unit'.
    unit : str, optional
        The unit of the lower and upper scale limits.
        Default: kpc
    rweight : float, optional
        Power-law exponent used to weight pairs by their separation.
        Default: None
    resolution : int, optional
        Number of radial logarithmic bin used to approximate the weighting by
        separation.
        Default: None
    zmin : float, optional
        Lowest redshift bin edge to generate (alternatively use 'edges').
        Default: None
    zmax : float, optional
        Highest redshift bin edge to generate (alternatively use 'edges').
        Default: None
    num_bins : int, optional
        Number of redshift bins to generate between 'zmin' and 'zmax'.
        Default: 30
    method : str, optional
        Method used to compute the spacing of bin edges.
        Default: linear
    edges : float, optional
        Use these custom bin edges instead of generating them.
        Default: None
    closed : str, optional
        String indicating the side of the bin intervals that are closed.
        Default: right
    max_workers : int, optional
        configure a custom maximum number of parallel workers to use
        Default: None
    verbose : str, optional
        lowest log level emitted by *yet_another_wizz*
        Default: info

    Returns
    -------
    YawCorrFuncHandle
        A handle for the `yaw.CorrFunc` instance that holds the pair counts.
    """

def yaw_summarize(**kwargs) -> Any:
    """
    A summarizer that computes a clustering redshift estimate from the measured
    correlation amplitudes.

    Evaluates the cross-correlation pair counts with the provided estimator.
    Additionally corrects for galaxy sample bias if autocorrelation measurements
    are provided as stage inputs.

    **Note:** This summarizer does not produce a PDF, but a ratio of
    correlation functions, which may result in negative values. Further
    modelling of the output is required.

    ---

    Compute a clustring redshift estimate and convert it to a PDF.

    ---

    This function was generated from the function
    rail.estimation.algos.cc_yaw.YawSummarize.summarize

    Parameters
    ----------
    cross_corr : CorrFunc, required
        Pair counts from the cross-correlation measurement, basis for the
        clustering redshift estimate.
    auto_corr_ref : CorrFunc, required
        Pair counts from the reference sample autocorrelation measurement,
        used to correct for the reference sample galaxy bias.
    auto_corr_unk : CorrFunc, required
        Pair counts from the unknown sample autocorrelation measurement,
        used to correct for the reference sample galaxy bias. Typically only
        availble when using simulated data sets.
    verbose : str, optional
        lowest log level emitted by *yet_another_wizz*
        Default: info

    Returns
    -------
    YawRedshiftDataHandle
        The clustering redshift estimate, spatial (jackknife) samples
        thereof, and its covariance matrix.
    """
