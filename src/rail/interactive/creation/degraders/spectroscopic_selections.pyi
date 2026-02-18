from typing import Any

def spec_selection(**kwargs) -> Any:
    """
    The super class of spectroscopic selections.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_BOSS(**kwargs) -> Any:
    """
    The class of spectroscopic selections with BOSS.

    BOSS selection function is based on
    http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php

    The selection has changed slightly compared to Dawson+13.

    BOSS covers an area of 9100 deg^2 with 893,319 galaxies.

    For BOSS selection, the data should at least include gri bands.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_BOSS.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_DEEP2(**kwargs) -> Any:
    """
    The class of spectroscopic selections with DEEP2.

    DEEP2 has a sky coverage of 2.8 deg^2 with ~53000 spectra.

    For DEEP2, one needs R band magnitude, B-R/R-I colors--which are not
    available for the time being, so we use LSST gri bands now. When the
    conversion degrader is ready, this subclass will be updated accordingly.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_DEEP2.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_DEEP2_LSST(**kwargs) -> Any:
    """
    The class of spectroscopic selections with DEEP2.

    Approximate Rubin->CFHT12K transforms based off of
    CWWSB SED colors

    B = g + 0.35 * (g-r)
    R = r - 0.3 * (r-i)
    I = i - 0.5 * (r-i)

    transform the cuts accordingly

    Also, original has
    B-R < 0.5
    modify to
    B-R < 0.33 to exclude a few more low-z galaxies
    leave speczSuccess unchanged from original implementation

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_DEEP2_LSST.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_DESI_BGS(**kwargs) -> Any:
    """
    The class of spectroscopic selections with DESI BGS .

    Implements a minimal DESI Bright Galaxy Survey (BGS) selection using:
      - r < 19.5

    Required bands in `data` (via config.colnames): r

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_DESI_BGS.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_ELG_LOP(**kwargs) -> Any:
    """
    The class of spectroscopic selections with DESI ELG LOP.

    Implements the simplified DESI ELG_LOP photometric selection using:
      - (g > 20) AND (gfib < 24.1)
      - 0.15 < (r − z)
      - (g − r) < 0.5 × (r − z) + 0.1
      - (g − r) < −1.2 × (r − z) + 1.3

    All of the above are combined with AND.

    Required bands in `data` (via config.colnames): g, r, z, gfib

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_DESI_ELG_LOP.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_DESI_LRG(**kwargs) -> Any:
    """
    The class of spectroscopic selections with DESI LRG (simplified).

    This implements a simplified DESI LRG photometric selection using:
      - zfiber < 21.60  (here approximated with z)
      - z − W1 > 0.8 × (r − z) − 0.6
      - (g − W1 > 2.9) OR (r − W1 > 1.8)
      - [ ((r − W1 > 1.8 × (W1 − 17.14)) AND (r − W1 > W1 − 16.33)) OR (r − W1 > 3.3) ]

    All of the above are combined with AND.

    Required bands in `data` (via config.colnames): g, r, z, W1

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_DESI_LRG.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_GAMA(**kwargs) -> Any:
    """
    The class of spectroscopic selections with GAMA.

    The GAMA survey covers an area of 286 deg^2, with ~238000 objects.

    The necessary column is r band.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_GAMA.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_HSC(**kwargs) -> Any:
    """
    The class of spectroscopic selections with HSC.

    For HSC, the data should at least include giz bands and redshift.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_HSC.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_VVDSf02(**kwargs) -> Any:
    """
    The class of spectroscopic selections with VVDSf02.

    It covers an area of 0.5 deg^2 with ~10000 sources.

    Necessary columns are i band magnitude and redshift.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_VVDSf02.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """

def spec_selection_zCOSMOS(**kwargs) -> Any:
    """
    The class of spectroscopic selections with zCOSMOS.

    It covers an area of 1.7 deg^2 with ~20000 galaxies.

    For zCOSMOS, the data should at least include i band and redshift.

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
    rail.creation.degraders.spectroscopic_selections.SpecSelection_zCOSMOS.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    N_tot : int, optional
        Number of selected sources
        Default: 10000
    nondetect_val : float, optional
        value to be removed for non detects
        Default: 99.0
    downsample : bool, optional
        If true, downsample the selected sources into a total number of N_tot
        Default: True
    success_rate_dir : str, optional
        The path to the directory containing success rate files.
        Default: rail.utils.path_utils.RAILDIR/rail/examples_data/creation_data/data/suc
        cess_rate_data
    percentile_cut : int, optional
        cut redshifts above this percentile
        Default: 100
    colnames : dict, optional
        a dictionary that includes necessary columns (magnitudes, colors and redshift)
        for selection. For magnitudes, the keys are ugrizy; for colors, the keys are,
        for example, gr standing for g-r; for redshift, the key is 'redshift'
        Default: {'u': 'mag_u_lsst', 'g': 'mag_g_lsst', 'r': 'mag_r_lsst', 'i':...}
    random_seed : int, optional
        random seed for reproducibility
        Default: 42

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """
