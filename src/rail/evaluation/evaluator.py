"""
Superstages corresponding to Evaluators of metrics, grouped by input types
"""

class ProbToProbEvaluator(RailStage):
   """ A stage for evaluating metrics of estimated PDFs and reference PDFs"""
    
    def __init__(self, args, comm=None):
        """Initialize as a RailStage object"""
        RailStage.__init__(self, args, comm=comm)

    def __init__(self, qp_ens_est, qp_ens_):
        """Class constructor.
        Parameters
        ----------
        qp_ens: qp.Ensemble object
            PDFs as qp.Ensemble
        """
        self._qp_ens = qp_ens

    def evaluate(self, data, truth):
        """Evaluate the performance of an estimator

        This will attach the input data and truth to this `Evaluator`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Parameters
        ----------
        data : qp.Ensemble
            The sample to evaluate
        truth : Table-like
            Table with the truth information

        Returns
        -------
        output : Table-like
            The evaluation metrics
        """

        self.set_data('input', data)
        self.set_data('truth', truth)
        self.run()
        self.finalize()
        return self.get_handle('output')

