"""
Superstages corresponding to Evaluators of metrics


"""

class Evaluator:
    """ A superclass for evaluating metrics"""
    def __init__(self, qp_ens):
        """Class constructor.
        Parameters
        ----------
        qp_ens: qp.Ensemble object
            PDFs as qp.Ensemble
        """
        self._qp_ens = qp_ens

    def evaluate(self):  #pragma: no cover
        """
        Evaluates the metric a function of the truth and prediction

        Returns
        -------
        metric: dictionary
            value of the metric and statistics thereof
        """
        raise NotImplementedError


