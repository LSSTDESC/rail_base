from typing import Any

def spec_selection_desi_phy(**kwargs) -> Any:
    """
    DESI tracer selector based on pre-computed redshift-dependent thresholds.

    Applies a selection to a simulation catalog by comparing a physical
    parameter column against a threshold that varies with redshift. The
    threshold table is provided externally (e.g. from abundance matching)
    and is not computed by this stage.

    All supported DESI tracer types (bgs, lrg, elg) select objects whose
    physical parameter value is *above* the redshift-interpolated threshold.

    Inputs
    ------
    input : PqHandle
        Simulation catalog containing the physical parameter column and a
        redshift column.
    threshold_table : TableHandle
        Table with two columns:
          - ``z``     : redshift bin centers
          - ``thresh``: threshold values at those redshift centers

    Output
    ------
    output : PqHandle
        Catalog after applying the DESI selection mask.

    ---

    Apply the DESI physical selection to a catalog.

    ---

    This function was generated from the function
    rail.creation.degraders.desi_selector_phy.SpecSelection_DESI_Phy.__call__

    Parameters
    ----------
    sample : table-like or PqHandle, required
        Input simulation catalog.
    drop_rows : bool, optional
        Drop selected rows from output table
        Default: True
    seed : unknown type, optional
        Set to an `int` to force reproducible results.
        Default: None
    desi_type : str, optional
        DESI tracer type: 'bgs', 'lrg', or 'elg'
        Default: lrg
    threshold_col : str, optional
        Column in the input catalog used for threshold-based selection (e.g.
        'log_peak_sub_halo_mass' for bgs/lrg, 'log_sfr' for elg)
        Default: None
    redshift_col : str, optional
        Column name for redshift in the input catalog
        Default: redshift
    threshold_table : str, optional
        Filename of the threshold file
        Default: None

    Returns
    -------
    pandas.core.frame.DataFrame
        Handle to the output catalog containing only the selected objects
        (when ``drop_rows=True``, the default) or the full catalog with a
        ``flag`` column (when ``drop_rows=False``).
    """
