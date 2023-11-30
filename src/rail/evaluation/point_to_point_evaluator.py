import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import PointToPointMetric

from rail.core.data import Hdf5Handle, TableHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator

# dynamically build a dictionary of all available metrics of the appropriate type
METRIC_DICT = {}
for subcls in PointToPointMetric.__subclasses__():
    METRIC_DICT[subcls.metric_name] = subcls

class PointToPointEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = 'PointToPointEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(list, [], required=False,
            msg="The metrics you want to evaluate."),
        chunk_size=Param(int, 1000, required=False,
            msg="The default number of PDFs to evaluate per loop."),
        _random_state=Param(float, default=None, required=False,
            msg="Random seed value to use for reproducible results."),
    )
    inputs = [('input', TableHandle),
              ('truth', TableHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        Evaluator.__init__(self, args, comm=comm)
        self._output_handle = None
        self._metric_dict = METRIC_DICT

    def run(self):
        print(f"Requested metrics: {self.config.metrics}")

        estimate_iterator = self.input_iterator('input')
        reference_iterator = self.input_iterator('truth')

        first = True
        for estimate_data_chunk, reference_data_chunk in zip(estimate_iterator, reference_iterator):
            chunk_start, chunk_end, estimate_data = estimate_data_chunk
            _, _, reference_data = reference_data_chunk

            print(f"Processing {self.rank} running evaluator on chunk {chunk_start} - {chunk_end}.")
            self._process_chunk(chunk_start, chunk_end, estimate_data, reference_data, first)
            first = False

        self._output_handle.finalize_write()

    def _process_chunk(self, start, end, estimate_data, reference_data, first):
        out_table = {}
        for metric in self.config.metrics:
            if metric not in self._metric_dict:
                #! Make the following a logged error instead of bailing out of the stage.
                # raise ValueError(
                # f"Unsupported metric requested: '{metric}'.
                # Available metrics are: {self._metric_dict.keys()}")
                continue

            this_metric = self._metric_dict[metric](**self.config.to_dict())
            out_table[metric] = this_metric.evaluate(estimate_data, reference_data)

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}

        if first:
            self._output_handle = self.add_handle('output', data=out_table_to_write)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)
