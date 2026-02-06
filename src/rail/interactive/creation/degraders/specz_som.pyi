from typing import Any

def som_spec_selector(**kwargs) -> Any:
    """
    Class that creates a specz sample by training a SOM on data with spec-z,
    classifying all galaxies from a larger sample via the SOM, then selecting
    the same number of galaxies in each SOM cell as there are in the specz
    sample.  If fewer galaxies are available in the large sample for a cell,
    it just takes as many as possible, so you can still mismatch the
    distribution numbers, i.e. if you have a lot of bright galaxies with
    speczs from a really wide survey like SDSS and the second dataset does
    not have the same areal coverage, then there may not be enough bright
    objects in the second dataset to select, so you will end up with fewer.

    For the columns used to construct the SOM, there are two sets of columns,
    `noncolor_cols` is a config option where you supply a list of columns that
    will be used directly in the SOM, e.g. redshift, i-magnitude, etc...
    `color_cols`, on the other hand, is a config parameter where the user
    supplies an ordered list of columns that will be differenced before being
    used as SOM inputs, e.g. if you supply ['u', 'g','r'] then a function in
    the code will compute u-g and g-r and use those in SOM construction.  The
    code combines the noncolor_cols and color_cols features and all are used
    in construction of the SOM.

    As this degrader inherits from `Selector`, it simply computes a mask, the
    Selector parent class code will perform the masking, and will return the
    final dataset that mimics the input reference sample.

    ---

    Entrypoint function for SomSpecSelector

    ---

    This function was generated from the function
    rail.creation.degraders.specz_som.SOMSpecSelector.__call__

    Parameters
    ----------
    input_data : TableLike, required
        The sample to be selected
    spec_data : TableLike, required
        A reference/spectroscopic data set
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    nondetect_val : float, optional
        value to be replaced with magnitude limit for non detects
        Default: 99.0
    noncolor_cols : list, optional
        data columns used for SOM, can be a single band ifyou will also be using
        colordata in 'color_cols', or can be as many as you want
        Default: ['i', 'redshift']
    noncolor_nondet : list, optional
        list of nondetect replacement values for the non-color cols
        Default: [28.62, -1.0]
    color_cols : list, optional
        columns that will be differenced to make colors.  This will be done in order, so
        put in increasing WL order
        Default: ['u', 'g', 'r', 'i', 'z', 'y']
    color_nondet : list, optional
        list of nondetect replacement vals for color columns
        Default: [27.79, 29.04, 29.06, 28.62, 27.98, 27.05]
    som_size : list, optional
        tuple containing the size (x, y) of the SOM
        Default: [32, 32]
    n_epochs : int, optional
        number of training epochs.
        Default: 10

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with selected sample
    """
