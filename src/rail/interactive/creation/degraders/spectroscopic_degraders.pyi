from typing import Any

def inv_redshift_incompleteness(**kwargs) -> Any:
    """
    Degrader that simulates incompleteness with a selection function inversely
    proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.

    ---

    The main interface method for ``Selector``.

    Adds noise to the input catalog

    This will attach the input to this `Selector`

    Then it will call the select() which add a flag column to the catalog. flag=1 means
    selected, 0 means dropped.

    If dropRows = True, the dropped rows will not be presented in the output catalog,
    otherwise, all rows will be presented.

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.spectroscopic_degraders.InvRedshiftIncompleteness.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    pivot_redshift : float, required
        redshift at which the incompleteness begins
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def line_confusion(**kwargs) -> Any:
    """
    Degrader that simulates emission line confusion.

    .. code-block:: python

       degrader = LineConfusion(true_wavelen=3727,
                                wrong_wavelen=5007,
                                frac_wrong=0.05)

    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift.

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.

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
    rail.creation.degraders.spectroscopic_degraders.LineConfusion.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    true_wavelen : float, required
        wavelength of the true emission line
    wrong_wavelen : float, required
        wavelength of the wrong emission line
    frac_wrong : float, required
        fraction of galaxies with confused emission lines
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """
