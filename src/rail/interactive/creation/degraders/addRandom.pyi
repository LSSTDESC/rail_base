from typing import Any

def add_column_of_random(**kwargs) -> Any:
    """
    Add a column of random numbers to a dataframe

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
    rail.creation.degraders.addRandom.AddColumnOfRandom.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.
    col_name : str, optional
        Name of the column with random numbers
        Default: chaos_bunny

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """
