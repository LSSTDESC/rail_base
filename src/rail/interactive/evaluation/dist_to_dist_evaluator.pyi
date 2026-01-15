from typing import Any

def dist_to_dist_evaluator(**kwargs) -> Any:
    """
    Evaluate the performance of a photo-z estimator against reference PDFs

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
    rail.evaluation.dist_to_dist_evaluator.DistToDistEvaluator.evaluate

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
    metric_integration_limits : list, optional
        The default end points for calculating metrics on a grid.
        Default: [0.0, 3.0]
    dx : float, optional
        The default step size when calculating metrics on a grid.
        Default: 0.01
    n_samples : int, optional
        The number of random samples to select for certain metrics.
        Default: 100

    Returns
    -------
    dict[str, DataHandle]
        The evaluation metrics
    """
