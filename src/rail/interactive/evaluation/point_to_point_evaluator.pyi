from typing import Any

def point_to_point_binned_evaluator(**kwargs) -> Any:
    """
    Evaluate the performance of a photo-z estimator against reference point estimate

    ---

    Evaluate the performance of an estimator

    This will attach the input data and truth to this `Evaluator`
    (for introspection and provenance tracking).

    Then it will call the run() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the data that it creates to this Estimator
    by using `self.add_data('output', output_data)`.

    ---

    This function was generated from the function
    rail.evaluation.point_to_point_evaluator.PointToPointBinnedEvaluator.evaluate

    Parameters
    ----------
    data : qp.Ensemble, required
        The sample to evaluate
    truth : Any, required
        Table with the truth information
    metrics : list, optional
        The metrics you want to evaluate.
        Default: []
    exclude_metrics : list, optional
        List of metrics to exclude
        Default: []
    metric_config : dict, optional
        configuration of individual_metrics
        Default: {}
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    seed : float, optional
        Random seed value to use for reproducible results.
        Default: None
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    reference_dictionary_key : str, optional
        The key in the `truth` dictionary where the redshift data is stored.
        Default: redshift
    point_estimate_key : str, optional
        The key in the point estimate table.
        Default: zmode
    bin_col : str, optional
        The column metrics are binned by
        Default: redshift
    bin_min : float, optional
        The mininum value of the binning edge
        Default: 0.0
    bin_max : float, optional
        The maximum value of the binning edge
        Default: 3.0
    nbin : int, optional
        The mininum value of the binning edge
        Default: 10

    Returns
    -------
    dict[str, DataHandle]
        The evaluation metrics
    """

def point_to_point_evaluator(**kwargs) -> Any:
    """
    Evaluate the performance of a photo-z estimator against reference point estimate

    ---

    Evaluate the performance of an estimator

    This will attach the input data and truth to this `Evaluator`
    (for introspection and provenance tracking).

    Then it will call the run() and finalize() methods, which need to
    be implemented by the sub-classes.

    The run() method will need to register the data that it creates to this Estimator
    by using `self.add_data('output', output_data)`.

    ---

    This function was generated from the function
    rail.evaluation.point_to_point_evaluator.PointToPointEvaluator.evaluate

    Parameters
    ----------
    data : qp.Ensemble, required
        The sample to evaluate
    truth : Any, required
        Table with the truth information
    metrics : list, optional
        The metrics you want to evaluate.
        Default: []
    exclude_metrics : list, optional
        List of metrics to exclude
        Default: []
    metric_config : dict, optional
        configuration of individual_metrics
        Default: {}
    chunk_size : int, optional
        Number of objects per chunk for parallel processing or to evalute per loop in
        single node processing
        Default: 10000
    seed : float, optional
        Random seed value to use for reproducible results.
        Default: None
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    reference_dictionary_key : str, optional
        The key in the `truth` dictionary where the redshift data is stored.
        Default: redshift
    point_estimate_key : str, optional
        The key in the point estimate table.
        Default: zmode

    Returns
    -------
    dict[str, DataHandle]
        The evaluation metrics
    """
