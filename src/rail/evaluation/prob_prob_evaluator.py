import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics import calculate_goodness_of_fit

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator

class ProbProbEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference PDFs"""

    name = 'ProbProbEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(list, [], msg="The metrics you want to evaluate."),
        chunk_size=Param(int, 1000, msg="The default number of PDFs to evaluate per loop.")
    )
    inputs = [('input', QPHandle),
              ('truth', QPHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        Evaluator.__init__(self, args, comm=comm)

    def run(self):
        out_table = {}
        print(f"Requested metrics: {self.config.metrics}")

        estimate_iterator = self.input_iterator('input')
        
        #! Correct the following line to be 'truth' !!!
        # comparison_iterator = self.input_iterator('input') 

        #! Need to implement something to ensure that the iterators are aligned.

        first = True
        for s, e, data in estimate_iterator:
            print(f"Processing {self.rank} running evaluator on chunk {s} - {e}.")
            self._process_chunk(s, e, data, first)
            first = False


        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}
        self.add_data('output', data=out_table_to_write)

    def _process_chunk(self, start, end, data, first):
        out_table = {}
        for metric in self.config.metrics:
            try:
                #! Fix the next line since it's just data vs. data !!!
                out_table[metric] = calculate_goodness_of_fit(data, data, fit_metric=metric)
            except KeyError:
                print(f"User requested unrecognized metric: {metric} - Skipping.")

        if first:
            # Need to determine how to correctly initialize this
            pass
        else:
            # Remove the "else", but figure out the correct "non-first" logic.
            pass
