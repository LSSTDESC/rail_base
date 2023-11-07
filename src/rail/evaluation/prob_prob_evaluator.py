import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics import calculate_goodness_of_fit, calculate_kld

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator

class ProbProbEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference PDFs"""

    name = 'ProbProbEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(list, [], msg="The metrics you want to evaluate."),
        chunk_size=Param(int, 1000, msg="The default number of PDFs to evaluate per loop."),
        limits=Param(tuple, (0.0, 3.0), msg="The default end points for calculating metrics on a grid."),
        dx=Param(float, 0.01, msg="The default step size when calculating metrics on a grid."),
    )
    inputs = [('input', QPHandle),
              ('truth', QPHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        Evaluator.__init__(self, args, comm=comm)
        self._output_handle = None

    def run(self):
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

        self._output_handle.finalize_write()

    def _process_chunk(self, start, end, data, first):
        out_table = {}
        for metric in self.config.metrics:
            try:
                #! Fix the next line since it's just data vs. data !!!
                out_table[metric] = calculate_goodness_of_fit(data, data, fit_metric=metric)
            except KeyError:
                print(f"User requested unrecognized metric: {metric} - Skipping.")
                if metric == 'kld':
                    out_table[metric] = calculate_kld(data, data, self.config.limits, self.config.dx)

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}

        if first:
            self._output_handle = self.add_handle('output', data=out_table_to_write)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)