from typing import Any

def quantity_cut(**kwargs) -> Any:
    """
    Degrader that applies a cut to the given columns.

    Note that if a galaxy fails any of the cuts on any one of its columns, that
    galaxy is removed from the sample.

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
    rail.creation.degraders.quantityCut.QuantityCut.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be selected
    cuts : dict, required
        Cuts to apply
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
