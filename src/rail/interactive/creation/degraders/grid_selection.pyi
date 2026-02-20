from typing import Any

def grid_selection(**kwargs) -> Any:
    """
    Uses the ratio of HSC spectroscpic galaxies to photometric galaxies to portion a
    sample into training and application samples. Option to implement a color-based
    redshift cut off in each pixel.
    Option of further degrading the training sample by limiting it to galaxies less than
    a redshift cutoff by specifying redshift_cut.

    .. code-block:: text

        color_redshift_cut: True or false, implements color-based redshift cut. Default
        is True.
            If True, ratio_file must include second key called 'data' with magnitudes,
            colors and spec-z from the spectroscopic sample.
        percentile_cut: If using color-based redshift cut, percentile in spec-z above
        which redshifts will be cut from training sample. Default is 99.0
        scaling_factor: Enables the user to adjust the ratios by this factor to change
        the overall number of galaxies kept.  For example, if you wish
            to generate 100,00 galaxies but only 50,000 are selected by default, then
            you can adjust factor up by a factor of 2 to return more galaixes.
        redshift_cut: redshift above which all galaxies will be removed from training
        sample. Default is 100
        ratio_file: hdf5 file containing an array of spectroscpic vs. photometric
        galaxies in each pixel. Default is hsc_ratios.hdf5 for an HSC based selection
        settings_file: pickled dictionary containing information about colors and
        magnitudes used in defining the pixels. Dictionary must include the following
        keys:
            'x_band_1': string, this is the band used for the magnitude in the color
            magnitude diagram. Default for HSC is 'i'.
            'x_band_2': string, this is the redder band used for the color in the color
            magnitude diagram.
            if x_band_2 string is not set to '' then the grid is assumed to be over
            color and x axis color is set to x_band_1 - x_band_2, default is ''.
            'y_band_1': string, this is the bluer band used for the color in the color
            magnitude grid. Default for HSC is 'g'.
            'y_band_2': string, this is the redder band used for the color in the color
            magnitude diagram.
            if y_band_2 is not set to '' then the y-band is assumed to be over color and
            is set to y_band_1 - y_band 2.
            'x_limits': 2-element list, this is a list of the lower and upper limits of
            the magnitude. Default for HSC is [13, 16],
            'y_limits': 2-element list, this is a list of the lower and upper limits of
            the color. Default for HSC is [-2, 6]}

    NOTE: the default 'HSC' grid file, located in
    rail/examples_data/creation_data/data/hsc_ratios_and_specz.hdf5, is based on data
    from the
    Second HSC Data Release, details of which can be found here:
    Aihara, H., AlSayyad, Y., Ando, M., et al. 2019, PASJ, 71, 114
    doi: 10.1093/pasj/psz103

    Update(Apr 16 2024): Now inherit from selector and implement the _select() instead
    of run()

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
    rail.creation.degraders.grid_selection.GridSelection.__call__

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
    color_redshift_cut : bool, optional
        using color-based redshift cut
        Default: True
    percentile_cut : float, optional
        percentile cut-off for each pixel in color-based redshift cut off
        Default: 99.0
    redshift_cut : float, optional
        cut redshifts above this value
        Default: 100.0
    ratio_file : str, optional
        path to ratio file
        Default: rail/examples_data/creation_data/data/hsc_ratios_and_specz.hdf5
    settings_file : str, optional
        path to pickled parameters file
        Default: rail/examples_data/creation_data/data/HSC_grid_settings.pkl
    random_seed : int, optional
        random seed for reproducibility
        Default: 12345
    scaling_factor : float, optional
        multiplicative factor for ratios to adjust number of galaxies kept
        Default: 1.588

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """
