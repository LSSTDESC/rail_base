import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import MetricOutputType
from qp.metrics.concrete_metric_classes import DistToDistMetric

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import BaseEvaluator


class DistToDistEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-z estimator against reference PDFs"""

    name = 'DistToDistEvaluator'
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(
        limits=Param(tuple, (0.0, 3.0), required=False,
            msg="The default end points for calculating metrics on a grid."),
        dx=Param(float, 0.01, required=False,
            msg="The default step size when calculating metrics on a grid."),
        num_samples=Param(int, 100, required=False,
            msg="The number of random samples to select for certain metrics."),
    )
    inputs = [('input', QPHandle),
              ('truth', QPHandle)]

    metric_base_class = DistToDistMetric
        
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
                centroids = this_metric.accumulate(
                    estimate_data,
                    reference_data,
                )
                if self.comm:
                    self._cached_data[metric] = centroids
                else:
                    if metric in self._cached_data:
                        self._cached_data[metric].append(centroids)
                    else:
                        self._cached_data[metric] = [centroids]

            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                print(f"{metric} with output type MetricOutputType.single_distribution not supported yet")
                continue

            else:
                out_table[metric] = this_metric.evaluate(estimate_data, reference_data)

        self._output_table_chunk_data(start, end, out_table, first)
