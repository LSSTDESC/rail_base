from typing import Any

def dereddener(**kwargs) -> Any:
    """
    Utility stage that does dereddening

    ---

    Return a converted table

    ---

    This function was generated from the function
    rail.tools.photometry_tools.Dereddener.__call__

    Parameters
    ----------
    data : table-like, required
        The data to be converted
    dustmap_dir : str, required
        Directory with dustmaps
    ra_name : str, optional
        Name of the RA column
        Default: ra
    dec_name : str, optional
        Name of the DEC column
        Default: dec
    mag_name : str, optional
        Template for the magnitude columns
        Default: mag_{band}_lsst
    band_a_env : dict, optional
        Reddening parameters
        Default: {'mag_u_lsst': 4.81, 'mag_g_lsst': 3.64, 'mag_r_lsst': 2.7,...}
    dustmap_name : str, optional
        Name of the dustmap in question
        Default: sfd
    copy_cols : list, optional
        Additional columns to copy
        Default: []
    copy_all_cols : bool, optional
        Copy all the columns
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        The converted version of the table
    """

def dust_map_base(**kwargs) -> Any:
    """
    Utility stage that does dereddening

    Note: set copy_all_cols=True to copy all
    columns in data, copy_cols will be ignored

    ---

    Return a converted table

    ---

    This function was generated from the function
    rail.tools.photometry_tools.DustMapBase.__call__

    Parameters
    ----------
    data : table-like, required
        The data to be converted
    dustmap_dir : str, required
        Directory with dustmaps
    ra_name : str, optional
        Name of the RA column
        Default: ra
    dec_name : str, optional
        Name of the DEC column
        Default: dec
    mag_name : str, optional
        Template for the magnitude columns
        Default: mag_{band}_lsst
    band_a_env : dict, optional
        Reddening parameters
        Default: {'mag_u_lsst': 4.81, 'mag_g_lsst': 3.64, 'mag_r_lsst': 2.7,...}
    dustmap_name : str, optional
        Name of the dustmap in question
        Default: sfd
    copy_cols : list, optional
        Additional columns to copy
        Default: []
    copy_all_cols : bool, optional
        Copy all the columns
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        The converted version of the table
    """

def hyperbolic_magnitudes(**kwargs) -> Any:
    """
    Convert a set of classical magnitudes to hyperbolic magnitudes  (Lupton et al.
    1999). Requires
    input from the initial stage (`HyperbolicSmoothing`) to supply optimal values for
    the smoothing
    parameters (b).

    ---

    Main method to call. Outputs hyperbolic magnitudes compuated from a set of smoothing
    parameters and input catalogue with classical magitudes and their respective errors.

    ---

    This function was generated from the function
    rail.tools.photometry_tools.HyperbolicMagnitudes.compute

    Parameters
    ----------
    data : `PqHandle`, required
        Input table with photometry (magnitudes or flux columns and their respective
        uncertainties) as defined by the configuration.
    parameters : `PqHandle`, required
        Table witdh smoothing parameters per photometric band, determined by
        `HyperbolicSmoothing`.
    value_columns : list, optional
        list of columns that prove photometric measurements (fluxes or magnitudes)
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    error_columns : list, optional
        list of columns with errors corresponding to value_columns (assuming same
        ordering)
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zeropoints : list, optional
        optional list of magnitude zeropoints for value_columns (assuming same ordering,
        defaults to 0.0)
        Default: []
    is_flux : bool, optional
        whether the provided quantities are fluxes or magnitudes
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        Output table containting hyperbolic magnitudes and their uncertainties. If the
        columns
        in the input table contain a prefix `mag_`, this output tabel will replace the
        prefix
        with `hyp_mag_`, otherwise the column names will be identical to the input
        table.
    """

def hyperbolic_smoothing(**kwargs) -> Any:
    """
    Initial stage to compute hyperbolic magnitudes (Lupton et al. 1999). Estimates the
    smoothing
    parameter b that is used by the second stage (`HyperbolicMagnitudes`) to convert
    classical to
    hyperbolic magnitudes.

    ---

    Main method to call. Computes the set of smoothing parameters (b) for an input
    catalogue
    with classical photometry and their respective errors. These parameters are required
    by the
    follow-up stage `HyperbolicMagnitudes` and are parsed as tabular data.

    ---

    This function was generated from the function
    rail.tools.photometry_tools.HyperbolicSmoothing.compute

    Parameters
    ----------
    data : `PqHandle`, required
        Input table with magnitude and magnitude error columns as defined in the
        configuration.
    value_columns : list, optional
        list of columns that prove photometric measurements (fluxes or magnitudes)
        Default: ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst',...]
    error_columns : list, optional
        list of columns with errors corresponding to value_columns (assuming same
        ordering)
        Default: ['mag_err_u_lsst', 'mag_err_g_lsst', 'mag_err_r_lsst',...]
    zeropoints : list, optional
        optional list of magnitude zeropoints for value_columns (assuming same ordering,
        defaults to 0.0)
        Default: []
    is_flux : bool, optional
        whether the provided quantities are fluxes or magnitudes
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        Table with smoothing parameters per photometric band and additional meta data.
    """

def lsst_flux_to_mag_converter(**kwargs) -> Any:
    """
    Utility stage that converts from fluxes to magnitudes

    Note, this is hardwired to take parquet files as input
    and provide hdf5 files as output

    ---

    Return a converted table

    ---

    This function was generated from the function
    rail.tools.photometry_tools.LSSTFluxToMagConverter.__call__

    Parameters
    ----------
    data : table-like, required
        The data to be converted
    bands : list, optional
        Names of the bands
        Default: ['u', 'g', 'r', 'i', 'z', 'y']
    flux_name : str, optional
        Template for band names
        Default: {band}_gaap1p0Flux
    flux_err_name : str, optional
        Template for band error column names
        Default: {band}_gaap1p0FluxErr
    mag_name : str, optional
        Template for magnitude column names
        Default: mag_{band}_lsst
    mag_err_name : str, optional
        Template for magnitude error column names
        Default: mag_err_{band}_lsst
    copy_cols : dict, optional
        Map of other columns to copy
        Default: {}
    mag_offset : float, optional
        Magntidue offset value
        Default: 31.4

    Returns
    -------
    pandas.core.frame.DataFrame
        The converted version of the table
    """

def reddener(**kwargs) -> Any:
    """
    Utility stage that does reddening

    ---

    Return a converted table

    ---

    This function was generated from the function
    rail.tools.photometry_tools.Reddener.__call__

    Parameters
    ----------
    data : table-like, required
        The data to be converted
    dustmap_dir : str, required
        Directory with dustmaps
    ra_name : str, optional
        Name of the RA column
        Default: ra
    dec_name : str, optional
        Name of the DEC column
        Default: dec
    mag_name : str, optional
        Template for the magnitude columns
        Default: mag_{band}_lsst
    band_a_env : dict, optional
        Reddening parameters
        Default: {'mag_u_lsst': 4.81, 'mag_g_lsst': 3.64, 'mag_r_lsst': 2.7,...}
    dustmap_name : str, optional
        Name of the dustmap in question
        Default: sfd
    copy_cols : list, optional
        Additional columns to copy
        Default: []
    copy_all_cols : bool, optional
        Copy all the columns
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        The converted version of the table
    """
