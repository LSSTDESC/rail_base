from typing import Any

def true_nz_histogrammer(**kwargs) -> Any:
    """
    Summarizer-like stage which simply histograms the true redshift

    ---

    The main interface method for ``TrueNZHistogrammer``.

    Creates histogram of N of Z_true.

    This will attach the sample to this `Stage` (for introspection and
    provenance tracking).

    Then it will call the run() and finalize() methods, which need to be
    implemented by the sub-classes.

    The run() method will need to register the data that it creates to this
    Estimator by using ``self.add_data('output', output_data)``.

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.estimation.algos.true_nz.TrueNZHistogrammer.histogram

    Parameters
    ----------
    catalog : TableLike, required
        The sample with the true NZ column
    tomo_bins : TableLike, required
        Tomographic bin assignemnets
    tomography_bins : A tablesio-compatible table, required
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    redshift_col : str, optional
        name of redshift column
        Default: redshift
    selected_bin : int, optional
        Which tomography bin to consider
        Default: -1
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a the histogram in QP format
    """
