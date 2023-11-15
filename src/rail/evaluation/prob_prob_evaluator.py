import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.concrete_metric_classes import DistToDistMetric

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator

# dynamically build a dictionary of all available metrics of the appropriate type
METRIC_DICT = {}
for subcls in DistToDistMetric.__subclasses__():
    METRIC_DICT[subcls.metric_name] = subcls

class ProbProbEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference PDFs"""

    name = 'ProbProbEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(list, [], required=False,
            msg="The metrics you want to evaluate."),
        chunk_size=Param(int, 1000, required=False,
            msg="The default number of PDFs to evaluate per loop."),
        limits=Param(tuple, (0.0, 3.0), required=False,
            msg="The default end points for calculating metrics on a grid."),
        dx=Param(float, 0.01, required=False,
            msg="The default step size when calculating metrics on a grid."),
        num_samples=Param(int, 100, required=False,
            msg="The number of random samples to select for certain metrics."),
        _random_state=Param(float, default=None, required=False,
            msg="Random seed value to use for reproducible results."),
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
            if metric not in METRIC_DICT:
                #! Make the following a logged error instead of bailing out of the stage.
                # raise ValueError(
                # f"Unsupported metric requested: '{metric}'.
                # Available metrics are: {METRIC_DICT.keys()}")
                continue

            this_metric = METRIC_DICT[metric](**self.config.to_dict())
            out_table[metric] = this_metric.evaluate(data, data)

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}

        if first:
            self._output_handle = self.add_handle('output', data=out_table_to_write)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)
