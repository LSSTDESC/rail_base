import numpy as np
from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import MetricOutputType
from qp.metrics.concrete_metric_classes import DistToDistMetric

from rail.core.data import QPHandle
from rail.evaluation.evaluator import BaseEvaluator


class DistToDistEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-z estimator against reference PDFs"""

    name = "DistToDistEvaluator"
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(
        limits=Param(
            tuple,
            (0.0, 3.0),
            required=False,
            msg="The default end points for calculating metrics on a grid.",
        ),
        dx=Param(
            float,
            0.01,
            required=False,
            msg="The default step size when calculating metrics on a grid.",
        ),
        num_samples=Param(
            int,
            100,
            required=False,
            msg="The number of random samples to select for certain metrics.",
        ),
    )
    inputs = [("input", QPHandle), ("truth", QPHandle)]

    metric_base_class = DistToDistMetric

    def _process_chunk(self, data_tuple, first):
        start = data_tuple[0]
        end = data_tuple[1]
        estimate_data = data_tuple[2]
        reference_data = data_tuple[3]

        out_table = {}

        for metric, this_metric in self._cached_metrics.items():
            if this_metric.metric_output_type == MetricOutputType.single_value:
                if not hasattr(this_metric, "accumulate"):
                    print(
                        f"{metric} with output type single_value does not support parallel processing yet"
                    )
                    continue

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
                if not hasattr(this_metric, 'accumulate'):
                    print(f"{metric} with output type MetricOutputType.single_value does not support parallel processing yet")
                    continue
                self._cached_metrics[metric] = this_metric
                self._cached_data[metric] = this_metric.accumulate(
                    estimate_data,
                    reference_data[self.config.reference_dictionary_key]
                )
            else:
                out_table[metric] = this_metric.evaluate(estimate_data, reference_data)

        self._output_table_chunk_data(start, end, out_table, first)

    def _process_all(self, data_tuple):
        estimate_data = data_tuple[0]
        reference_data = data_tuple[1]

        out_table = {}
        summary_table = {}
        single_distribution_summary = {}

        for metric, this_metric in self._cached_metrics.items():
            if metric not in self._metric_dict:
                print(
                    f"Unsupported metric requested: '{metric}'.  "
                    "Available metrics are: {self._metric_dict.keys()}"
                )
                continue

            metric_result = this_metric.evaluate(estimate_data, reference_data)

            if this_metric.metric_output_type == MetricOutputType.single_value:
                summary_table[metric] = metric_result
            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                single_distribution_summary[this_metric.metric_name] = metric_result
            else:
                out_table[metric] = metric_result

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}
        self._output_handle = self.add_handle('output', data=out_table_to_write)
        self._summary_handle = self.add_handle('summary', data=summary_table)
        self._single_distribution_summary_handle = self.add_handle('single_distribution_summary', data=single_distribution_summary)
