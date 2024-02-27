import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import MetricOutputType
from qp.metrics.concrete_metric_classes import DistToPointMetric

from rail.core.data import Hdf5Handle, QPHandle, TableHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import BaseEvaluator


class DistToPointEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = 'DistToPointEvaluator'
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(
        limits=Param(tuple, (0.0, 3.0), required=False,
            msg="The default end points for calculating metrics on a grid."),
        dx=Param(float, 0.01, required=False,
            msg="The default step size when calculating metrics on a grid."),
        quantile_grid=Param(list, np.linspace(0,1,100), required=False,
            msg="The quantile value grid on which to evaluate the CDF values. (0, 1)"),
        x_grid=Param(list, np.linspace(0,2.5, 301), required=False,
            msg="The x-value grid at which to evaluate the pdf values."),
        hdf5_groupname=Param(str, "photometry", required=False,
            msg="HDF5 Groupname for truth table."),
        reference_dictionary_key=Param(str, "redshift", required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored."),
    )
    inputs = [('input', QPHandle),
              ('truth', TableHandle)]

    metric_base_class = DistToPointMetric

    def _process_chunk(self, data_tuple, first):

        start = data_tuple[0]
        end = data_tuple[1]
        estimate_data = data_tuple[2]
        reference_data = data_tuple[3]

        out_table = {}
        for metric in self.config.metrics:

            if metric not in self._metric_dict:
                #! Make the following a logged error instead of bailing out of the stage.
                # raise ValueError(
                # f"Unsupported metric requested: '{metric}'.
                # Available metrics are: {self._metric_dict.keys()}")
                continue

            this_metric = self._metric_dict[metric](**self.config.to_dict())

            if this_metric.metric_output_type == MetricOutputType.single_value:
                if not hasattr(this_metric, 'accumulate'):
                    print(f"{metric} with output type MetricOutputType.single_value does not support parallel processing yet")
                    continue
                self._cached_metrics[metric] = this_metric
                self._cached_data[metric] = this_metric.accumulate(
                    estimate_data,
                    reference_data[self.config.reference_dictionary_key]
                )
            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                print(f"{metric} with output type MetricOutputType.single_distribution not supported yet")
                continue
            else:
                self._cached_metrics[metric] = this_metric
                out_table[metric] = this_metric.evaluate(
                    estimate_data,
                    reference_data[self.config.reference_dictionary_key]
                )
                
        self._output_table_chunk_data(start, end, out_table, first)
