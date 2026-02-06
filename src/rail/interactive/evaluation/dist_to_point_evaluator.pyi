from typing import Any

def dist_to_point_evaluator(**kwargs) -> Any:
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
    rail.evaluation.dist_to_point_evaluator.DistToPointEvaluator.evaluate

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
    metric_integration_limits : list, optional
        The default end points for calculating metrics on a grid.
        Default: [0.0, 3.0]
    dx : float, optional
        The default step size when calculating metrics on a grid.
        Default: 0.01
    quantile_grid : list, optional
        The quantile value grid on which to evaluate the CDF values. (0, 1)
        Default: [0.0, 0.010101010101010102, 0.020202020202020204,...]
    x_grid : list, optional
        The x-value grid at which to evaluate the pdf values.
        Default: [0.0, 0.008333333333333333, 0.016666666666666666, 0.025,...]
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default: photometry
    reference_dictionary_key : str, optional
        The key in the `truth` dictionary where the redshift data is stored.
        Default: redshift

    Returns
    -------
    dict[str, DataHandle]
        The evaluation metrics
    """
