from typing import Any

def point_est_hist_informer(**kwargs) -> Any:
    """
    Placeholder Informer

    ---

    The main interface method for Informers

    This will attach the input_data to this `Informer`
    (for introspection and provenance tracking).

    Then it will call the run(), validate() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the model that it creates to this Estimator
    by using `self.add_data('model', model)`.

    Finally, this will return a ModelHandle providing access to the trained model.

    ---

    This function was generated from the function
    rail.estimation.algos.point_est_hist.PointEstHistInformer.inform

    Parameters
    ----------
    training_data : qp.Ensemble | str, optional
        Per-galaxy p(z), and any ancilary data associated with it, by default "None"
    truth_data : TableLike | str, optional
        Table with the true redshifts, by default "None"
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000

    Returns
    -------
    dict[str, ModelHandle]
        Handle providing access to trained model
    """

def point_est_hist_masked_summarizer(**kwargs) -> Any:
    """
    Summarizer which simply histograms a point estimate

    ---

    Override the Summarizer.summarize() method to take tomo bins
    as an additional input

    ---

    This function was generated from the function
    rail.estimation.algos.point_est_hist.PointEstHistMaskedSummarizer.summarize

    Parameters
    ----------
    input : qp.Ensemble
        Per-galaxy p(z), and any ancilary data associated with it
    tomo_bins : TableLike | None, optional
        Tomographic bins file, by default None
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    seed : int, optional
        random seed
        Default: 87
    point_estimate_key : str, optional
        Which point estimate to use
        Default: zmode
    n_samples : int, optional
        Number of sample distributions to return
        Default: 1000
    selected_bin : int, optional
        bin to use
        Default: -1

    Returns
    -------
    qp.core.ensemble.Ensemble
        Ensemble with n(z), and any ancilary data
    """

def point_est_hist_summarizer(**kwargs) -> Any:
    """
    Summarizer which simply histograms a point estimate

    ---

    The main run method for the summarization, should be implemented
    in the specific subclass.

    This will attach the input_data to this `PZtoNZSummarizer`
    (for introspection and provenance tracking).

    Then it will call the run() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the data that it creates to this Estimator
    by using `self.add_data('output', output_data)`.

    Finally, this will return a QPHandle providing access to that output data.

    ---

    This function was generated from the function
    rail.estimation.algos.point_est_hist.PointEstHistSummarizer.summarize

    Parameters
    ----------
    input : qp.Ensemble
        Per-galaxy p(z), and any ancillary data associated with it
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    seed : int, optional
        random seed
        Default: 87
    point_estimate_key : str, optional
        Which point estimate to use
        Default: zmode
    n_samples : int, optional
        Number of sample distributions to return
        Default: 1000

    Returns
    -------
    qp.core.ensemble.Ensemble
        Ensemble with n(z), and any ancillary data
    """
