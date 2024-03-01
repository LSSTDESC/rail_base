import numpy as np
import qp
from ceci.config import StageParameter as Param
from qp.metrics.base_metric_classes import (
    MetricOutputType,
    PointToPointMetric,
)

from rail.core.data import TableHandle, QPHandle
from rail.evaluation.evaluator import BaseEvaluator


class PointToPointEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "PointToPointEvaluator"
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(
        hdf5_groupname=Param(
            str, "photometry", required=False, msg="HDF5 Groupname for truth table."
        ),
        reference_dictionary_key=Param(
            str,
            "redshift",
            required=False,
            msg="The key in the `truth` dictionary where the redshift data is stored.",
        ),
        point_estimate_key=Param(
            str, "zmode", required=False, msg="The key in the point estimate table."
        ),
    )
    inputs = [("input", QPHandle), ("truth", TableHandle)]

    metric_base_class = PointToPointMetric

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
                    np.squeeze(estimate_data.ancil[self.config.point_estimate_key]),
                    reference_data[self.config.reference_dictionary_key],
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
                out_table[metric] = this_metric.evaluate(
                    np.squeeze(estimate_data.ancil[self.config.point_estimate_key]),
                    reference_data[self.config.reference_dictionary_key],
                )

        self._output_table_chunk_data(start, end, out_table, first)

    def _process_all(self, data_tuple):
        estimate_data = data_tuple[0]
        reference_data = data_tuple[1][self.config.hdf5_groupname]

        out_table = {}
        summary_table = {}
        single_distribution_summary = qp.Ensemble(qp.stats.norm, data=dict(loc=[], scale=[]))

        for metric in self.config.metrics:
            if metric not in self._metric_dict:
                #! Make the following a logged error instead of bailing out of the stage.
                print(
                    f"Unsupported metric requested: '{metric}'.  "
                    "Available metrics are: {self._metric_dict.keys()}"
                )
                continue

            this_metric = self._metric_dict[metric](**self.config.get(metric, {}))

            if this_metric.metric_output_type == MetricOutputType.single_value:
                summary_table[metric] = np.array(
                    [
                        this_metric.evaluate(
                            np.squeeze(
                                estimate_data.ancil[self.config.point_estimate_key]
                            ),
                            reference_data[self.config.reference_dictionary_key],
                        )
                    ]
                )

            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                single_distribution_ensemble = this_metric.finalize(self._cached_data[metric])
                single_distribution_ensemble.set_ancil({'metric': [metric.name]})

                # append the ensembles into a single output ensemble
                if single_distribution_summary is None:
                    single_distribution_summary = single_distribution_ensemble
                else:
                    single_distribution_summary.append(single_distribution_ensemble)
            else:
                out_table[metric] = this_metric.evaluate(
                    np.squeeze(estimate_data.ancil[self.config.point_estimate_key]),
                    reference_data[self.config.reference_dictionary_key],
                )

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}
        self._output_handle = self.add_handle('output', data=out_table_to_write)
        self._summary_handle = self.add_handle('summary', data=summary_table)
        self._single_distribution_summary_handle = self.add_handle('single_distribution_summary', data=single_distribution_summary)
