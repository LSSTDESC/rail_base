from typing import Any

def obs_condition(**kwargs) -> Any:
    """
    Photometric errors based on observation conditions

    This degrader calculates spatially-varying photometric errors
    using input survey condition maps. The error is based on the
    LSSTErrorModel from the PhotErr python package.

    .. code-block:: text

        mask: str, optional
            Path to the mask covering the survey
            footprint in HEALPIX format. Notice that
            all negative values will be set to zero.
        weight: str, optional
            Path to the weights HEALPIX format, used
            to assign sample galaxies to pixels. Default
            is weight="", which uses uniform weighting.
            tot_nVis_flag: bool, optional
            If any map for nVisYr are provided, this flag
            indicates whether the map shows the total number of
            visits in nYrObs (tot_nVis_flag=True), or the average
            number of visits per year (tot_nVis_flag=False). The
            default is set to True.
        map_dict: dict, optional
            A dictionary that contains the paths to the
            survey condition maps in HEALPIX format. This dictionary
            uses the same arguments as LSSTErrorModel (from PhotErr).
            The following arguments, if supplied, may contain either
            a single number (as in the case of LSSTErrorModel), or a path:
            [m5, nVisYr, airmass, gamma, msky, theta, km, tvis, EBV]
            For the following keys:
            [m5, nVisYr, gamma, msky, theta, km]
            numbers/paths for specific bands must be passed.
            Example:
            {"m5": {"u": path, ...}, "theta": {"u": path, ...},}
            Other LSSTErrorModel parameters can also be passed
            in this dictionary (e.g. a necessary one may be [nYrObs]
            or the survey condition maps).
            If any argument is not passed, the default value in
            PhotErr's LsstErrorModel is adopted.

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.observing_condition_degrader.ObsCondition.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.
    nside : int, optional
        nside for the input maps in HEALPIX format.
        Default: 128
    mask : str, optional
        mask for the input maps in HEALPIX format.
        Default: rail/examples_data/creation_data/data/survey_conditions/DC2-mask-neg-
        nside-128.fits
    weight : str, optional
        weight for assigning pixels to galaxies in HEALPIX format.
        Default: rail/examples_data/creation_data/data/survey_conditions/DC2-dr6-
        galcounts-i20-i25.3-nside-128.fits
    tot_nVis_flag : bool, optional
        flag indicating whether nVisYr is the total or average per year if supplied.
        Default: True
    random_seed : int, optional
        random seed for reproducibility
        Default: 42
    map_dict : dict, optional
        dictionary containing the paths to the survey condition maps and/or additional
        LSSTErrorModel parameters.
        Default: {'m5': {'i':...}

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """
