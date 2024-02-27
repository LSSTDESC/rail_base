import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import MetricOutputType, PointToPointMetric
from qp.metrics import point_estimate_metric_classes

from rail.core.data import Hdf5Handle, TableHandle, QPHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import BaseEvaluator


class PointToPointEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = 'PointToPointEvaluator'
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(
        reference_dictionary_key=Param(str, "redshift", required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored."),
        point_estimate_key=Param(str, "redshift", required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored."),                        
    )
    inputs = [('input', QPHandle),
              ('truth', TableHandle)]

    metric_base_class = PointToPointMetric

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
                print(f"Unsupported metric requested: '{metric}'.  Available metrics are: {self._metric_dict.keys()}")
                continue

            this_metric = self._metric_dict[metric](**self.config.get(metric, {}))
            
            if this_metric.metric_output_type == MetricOutputType.single_value:
                if not hasattr(this_metric, 'accumulate'):
                    print(f"{metric} with output type MetricOutputType.single_value does not support parallel processing yet")
                    continue
                
                self._cached_metrics[metric] = this_metric
                centroids = this_metric.accumulate(
                    np.squeeze(estimate_data.ancil[self.config.point_estimate_key]),
                    reference_data[self.config.reference_dictionary_key]
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
                out_table[metric] = this_metric.evaluate(
                    np.squeeze(estimate_data.ancil[self.config.point_estimate_key]),
                    reference_data[self.config.reference_dictionary_key]
                )

        self._output_table_chunk_data(start, end, out_table, first)

