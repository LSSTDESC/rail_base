"""
" Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics import create_metric, list_metrics, MetricInputType, MetricOutputType
from qp.metrics.base_metric_classes import BaseMetric

from rail.core.data import Hdf5Handle, QPOrTableHandle
from rail.core.stage import RailStage
from rail.core.common_params import SHARED_PARAMS
from rail.evaluation.evaluator import BaseEvaluator


class SingleEvaluator(BaseEvaluator):
    """Evaluate the performance of a photo-Z estimator """

    name = 'SingleEvaluator'
    config_options = BaseEvaluator.config_options.copy()
    config_options.update(exclude_metrics=Param(list, msg="List of metrics to exclude"),
                          point_estimates=Param(list, msg="List of point estimates to use", default=[]),
                          metric_config=Param(dict, msg="configuration of individual_metrics", default={}),
                          truth_point_estimates=Param(
                              list, msg="List of true point values to use", default=[])
                          )
    inputs = [('input', QPOrTableHandle),
              ('truth', QPOrTableHandle)]
    outputs = [('output', Hdf5Handle),
               ('cache', Hdf5Handle)]

    metric_base_class = BaseMetric

    def __init__(self, args, comm=None):
        """Initialize Evaluator"""
        RailStage.__init__(self, args, comm=comm)
        self._input_data_type = QPOrTableHandle.PdfOrValue.unknown
        self._truth_data_type = QPOrTableHandle.PdfOrValue.unknown

    def evaluate(self, data, truth):
        """Evaluate the performance of an estimator

        This will attach the input data and truth to this `Evaluator`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Parameters
        ----------
        data : qp.Ensemble
            The sample to evaluate
        truth : Table-like
            Table with the truth information

        Returns
        -------
        output : Table-like
            The evaluation metrics
        """
        self.set_data('input', data)
        self.set_data('truth', truth)
        self.run()
        self.finalize()
        return self.get_handle('output')

    def run(self):  # pylint: disable=too-many-branches
        """ Run method

        Evaluate all the metrics and put them into a table

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Get the truth data from the data store under this stages 'truth' tag
        Puts the data into the data store under this stages 'output' tag
        """
        input_data_handle = self.get_handle('input')
        truth_data_handle = self.get_handle('truth')

        self._input_data_type = input_data_handle.check_pdf_or_point()
        self._truth_data_type = truth_data_handle.check_pdf_or_point()

        BaseEvaluator.run(self)
        

    def _process_chunk(self, data_tuple, first)
        
        out_data = {}

        start = data_tuple[0]
        end = data_tuple[1]
        input_data = data_tuple[2]
        truth_data = data_tuple[3]

        print(start, end)

        for metric, this_metric in self._cached_metrics.items():
            if this_metric.metric_input_type == MetricInputType.single_ensemble:
                if not self._input_data_type.has_dist():
                    continue
                key_val = f"{metric}"
                try:
                    metric_chunk_data[key_val] = this_metric.evaluate(input_data)
                except Exception:
                    print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
            elif this_metric.metric_input_type == MetricInputType.dist_to_dist:
                if not self._input_data_type.has_dist() or not self._truth_data_type.has_dist():
                    continue
                key_val = f"{metric}"
                try:
                    metric_chunk_data[key_val] = this_metric.evaluate(input_data, truth_data)
                except Exception:
                    print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
            elif this_metric.metric_input_type == MetricInputType.dist_to_point:
                if not self._input_data_type.has_dist() or not self._truth_data_type.has_point():
                    continue
                for truth_point_estimate_ in self.config.truth_point_estimates:
                    key_val = f"{metric}_{truth_point_estimate_}"
                    print(key_val)
                    try:
                        ret_data = this_metric.evaluate(
                            input_data,
                            truth_data[truth_point_estimate_],
                        )
                        metric_chunk_data[key_val] = ret_data
                    except Exception:
                        print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
            elif this_metric.metric_input_type == MetricInputType.point_to_point:
                if not self._input_data_type.has_point() or not self._truth_data_type.has_point():
                    continue
                for point_estimate_ in self.config.point_estimates:
                    key_val = f"{metric}_{point_estimate_}_{truth_point_estimate_}"
                    print(key_val)
                    point_data = input_data.ancil[point_estimate_]
                    for truth_point_estimate_ in self.config.truth_point_estimates:
                        try:
                            ret_data = this_metric.evaluate(
                                point_data,
                                truth_data[truth_point_estimate_],
                            )
                            metric_chunk_data[key_val] = ret_data
                        except Exception:
                            print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
            elif this_metric.metric_input_type == MetricInputType.point_to_dist:
                if not self._input_data_type.has_point() or not self._truth_data_type.has_dist():
                    continue
                for point_estimate_ in self.config.point_estimates:
                    key_val = f"{metric_name}_{point_estimate_}"
                    print(key_val)
                    try:
                        metric_chunk_data[key_val] = metric_evaluator_.evaluate(
                            input_data,
                            truth_data,
                        )
                    except Exception:
                        print(f"Failed to evaluate {key_val} for chunk {start}:{end}")

            if this_metric.metric_output_type == MetricOutputType.single_value:
                for key, val in metric_chunk_data.items():
                    metric_chunk_data[key] = np.array([val])
                print(f"caching {metric_name}")
                cached_data.update(metric_chunk_data)
            elif metric_evaluator_.metric_output_type == MetricOutputType.one_value_per_distribution:
                print(f"saving {metric_name}")
                out_data.update(metric_chunk_data)
            elif metric_evaluator_.metric_output_type == MetricOutputType.single_distribution:
                print(f"dropping {metric_name}")

            self._do_chunk_output(cached_data, out_data, start, end, first)

        print('finalize')

        for metric_evaluator_ in metric_evaluators:
            metric_evaluator_.finalize()
        self._finalize_run()
        print('done')

  
