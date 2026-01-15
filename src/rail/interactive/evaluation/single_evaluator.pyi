from typing import Any

def single_evaluator(**kwargs) -> Any:
    """
    Evaluate the performance of a photo-Z estimator

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
    rail.evaluation.single_evaluator.SingleEvaluator.evaluate

    Parameters
    ----------
    input : dict["data": qp.Ensemble, "truth": Any]
        Dictionary of input data with the following keys:
        data: qp.Ensemble - The sample to evaluate
        truth: Any - Table with the truth information
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
    point_estimates : list, optional
        List of point estimates to use
        Default: []
    truth_point_estimates : list, optional
        List of true point values to use
        Default: []
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry

    Returns
    -------
    dict[str, DataHandle]
        The evaluation metrics
    """
