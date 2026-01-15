from typing import Any

def old_evaluator(**kwargs) -> Any:
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
    rail.evaluation.evaluator.OldEvaluator.evaluate

    Parameters
    ----------
    input : dict["data": qp.Ensemble, "truth": Any]
        Dictionary of input data with the following keys:
        data: qp.Ensemble - The sample to evaluate
        truth: Any - Table with the truth information
    zmin : float, optional
        The minimum redshift of the z grid or sample
        Default: 0.0
    zmax : float, optional
        The maximum redshift of the z grid or sample
        Default: 3.0
    nzbins : int, optional
        The number of gridpoints in the z grid
        Default: 301
    pit_metrics : str, optional
        PIT-based metrics to include
        Default: all
    point_metrics : str, optional
        Point-estimate metrics to include
        Default: all
    hdf5_groupname : str, optional
        name of hdf5 group for data, if None, then set to ''
        Default:
    do_cde : bool, optional
        Evaluate CDE Metric
        Default: True
    redshift_col : str, optional
        name of redshift column
        Default: redshift

    Returns
    -------
    dict
        The evaluation metrics
    """
