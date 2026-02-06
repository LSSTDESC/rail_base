from typing import Any

def naive_stack_informer(**kwargs) -> Any:
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
    rail.estimation.algos.naive_stack.NaiveStackInformer.inform

    Parameters
    ----------
    training_data : qp.Ensemble | str, required
        Per-galaxy p(z), and any ancilary data associated with it, by default "None"
    truth_data : TableLike | str, required
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

def naive_stack_masked_summarizer(**kwargs) -> Any:
    """
    Stage NaiveStackMaskedSummarizer

    ---

    Override the Summarizer.summarize() method to take tomo bins
    as an additional input

    ---

    This function was generated from the function
    rail.estimation.algos.naive_stack.NaiveStackMaskedSummarizer.summarize

    Parameters
    ----------
    input_data : qp.Ensemble, required
        Per-galaxy p(z), and any ancillary data associated with it
    tomography_bins : A tablesio-compatible table, required
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
    n_samples : int, optional
        Number of sample distributions to create
        Default: 1000
    selected_bin : int, optional
        bin to use
        Default: -1

    Returns
    -------
    qp.core.ensemble.Ensemble
        Ensemble with n(z), and any ancillary data
    """

def naive_stack_summarizer(**kwargs) -> Any:
    """
    Summarizer which stacks individual P(z)

    ---

    Summarizer for NaiveStack which returns multiple items

    ---

    This function was generated from the function
    rail.estimation.algos.naive_stack.NaiveStackSummarizer.summarize

    Parameters
    ----------
    input_data : qp.Ensemble, required
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
    n_samples : int, optional
        Number of sample distributions to create
        Default: 1000

    Returns
    -------
    QPHandle | dict[str, QPHandle]
        Ensemble with n(z), and any ancillary data
        Return type depends on `output_mode`
    """
