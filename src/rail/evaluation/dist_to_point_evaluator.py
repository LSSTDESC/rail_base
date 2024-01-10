import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.concrete_metric_classes import DistToPointMetric

from rail.core.data import Hdf5Handle, QPHandle, TableHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator

# dynamically build a dictionary of all available metrics of the appropriate type
METRIC_DICT = {}
for subcls in DistToPointMetric.__subclasses__():
    METRIC_DICT[subcls.metric_name] = subcls

class DistToPointEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = 'DistToPointEvaluator'
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
        quantile_grid=Param(list, np.linspace(0,1,100), required=False,
            msg="The quantile value grid on which to evaluate the CDF values. (0, 1)"),
        x_grid=Param(list, np.linspace(0,2.5, 301), required=False,
            msg="The x-value grid at which to evaluate the pdf values."),
        _random_state=Param(float, default=None, required=False,
            msg="Random seed value to use for reproducible results."),
        hdf5_groupname=Param(str, "photometry", required=False,
            msg="HDF5 Groupname for truth table."),
        reference_dictionary_key=Param(str, "redshift", required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored."),
    )
    inputs = [('input', QPHandle),
              ('truth', TableHandle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        Evaluator.__init__(self, args, comm=comm)
        self._output_handle = None
        self._metric_dict = METRIC_DICT

    def run(self):
        print(f"Requested metrics: {self.config.metrics}")

        estimate_iterator = self.input_iterator('input', groupname=None)
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
            out_table[metric] = this_metric.evaluate(
                estimate_data, reference_data[self.config.reference_dictionary_key]
            )

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}

        if first:
            self._output_handle = self.add_handle('output', data=out_table_to_write)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)
