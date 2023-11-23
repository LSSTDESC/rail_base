"""
" Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

import numpy as np

from ceci.config import StageParameter as Param
from qp.metrics import create_metric, list_metrics, MetricInputType, MetricOutputType

from rail.core.data import Hdf5Handle, QPOrTableHandle
from rail.core.stage import RailStage
from rail.core.common_params import SHARED_PARAMS


class SingleEvaluator(RailStage):
    """Evaluate the performance of a photo-Z estimator """

    name = 'SingleEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(redshift_col=SHARED_PARAMS,
                          metrics=Param(list, msg="List of metrics to evaluate"),
                          exclude_metrics=Param(list, msg="List of metrics to exclude"),
                          point_estimates=Param(list, msg="List of point estimates to use", default=[]),
                          metric_config=Param(dict, msg="configuration of individual_metrics", default={}),
                          truth_point_estimates=Param(
                              list, msg="List of true point values to use", default=[])
                          )
    inputs = [('input', QPOrTableHandle),
              ('truth', QPOrTableHandle)]
    outputs = [('output', Hdf5Handle),
               ('cache', Hdf5Handle)]


    def __init__(self, args, comm=None):
        """Initialize Evaluator"""
        RailStage.__init__(self, args, comm=comm)
        self._input_data_type = QPOrTableHandle.PdfOrValue.unknown
        self._truth_data_type = QPOrTableHandle.PdfOrValue.unknown
        self._output_array_handle = None
        self._output_cache_handle = None
        self._output_summary_handle = None
        self._metric_config_dict = {}

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

        self._build_config_dict()

        metric_evaluators = [
            create_metric(key, **val) for key, val in self._metric_config_dict.items()
        ]

        # This is to set up the output handle
        itr = self._setup_iterator([input_data_handle.tag, truth_data_handle.tag])
        self._initialize_run()

        for metric_evaluator_ in metric_evaluators:
            metric_evaluator_.initialize()

        self._initialize_run()

        first = True
        for data_chunks in itr:  # pylint: disable=too-many-nested-blocks
            out_data = {}
            cached_data = {}

            start = data_chunks[0]
            end = data_chunks[1]
            input_data = data_chunks[2]
            truth_data = data_chunks[3]

            print(start, end)

            for metric_evaluator_ in metric_evaluators:

                metric_chunk_data = {}
                metric_name = metric_evaluator_.metric_name
                print(metric_name, metric_evaluator_.metric_input_type, metric_evaluator_.metric_output_type)

                if metric_evaluator_.metric_input_type == MetricInputType.single_ensemble:
                    if not self._input_data_type.has_dist():
                        continue
                    key_val = f"{metric_name}"
                    try:
                        metric_chunk_data[key_val] = metric_evaluator_.evaluate(input_data)
                    except Exception:
                        print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
                elif metric_evaluator_.metric_input_type == MetricInputType.dist_to_dist:
                    if not self._input_data_type.has_dist() or not self._truth_data_type.has_dist():
                        continue
                    key_val = f"{metric_name}"
                    try:
                        metric_chunk_data[key_val] = metric_evaluator_.evaluate(input_data, truth_data)
                    except Exception:
                        print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
                elif metric_evaluator_.metric_input_type == MetricInputType.dist_to_point:
                    if not self._input_data_type.has_dist() or not self._truth_data_type.has_point():
                        continue
                    for truth_point_estimate_ in self.config.truth_point_estimates:
                        key_val = f"{metric_name}_{truth_point_estimate_}"
                        print(key_val)
                        try:
                            ret_data = metric_evaluator_.evaluate(
                                input_data,
                                truth_data[truth_point_estimate_],
                            )
                            metric_chunk_data[key_val] = ret_data
                        except Exception:
                            print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
                elif metric_evaluator_.metric_input_type == MetricInputType.point_to_point:
                    if not self._input_data_type.has_point() or not self._truth_data_type.has_point():
                        continue
                    for point_estimate_ in self.config.point_estimates:
                        key_val = f"{metric_name}_{point_estimate_}_{truth_point_estimate_}"
                        print(key_val)
                        point_data = input_data.ancil[point_estimate_]
                        for truth_point_estimate_ in self.config.truth_point_estimates:
                            try:
                                ret_data = metric_evaluator_.evaluate(
                                    point_data,
                                    truth_data[truth_point_estimate_],
                                )
                                metric_chunk_data[key_val] = ret_data
                            except Exception:
                                print(f"Failed to evaluate {key_val} for chunk {start}:{end}")
                elif metric_evaluator_.metric_input_type == MetricInputType.point_to_dist:
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
                if metric_evaluator_.metric_output_type == MetricOutputType.single_value:
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
            first = False

        print('finalize')

        for metric_evaluator_ in metric_evaluators:
            metric_evaluator_.finalize()
        self._finalize_run()
        print('done')

    def _build_config_dict(self):
        """Build the configuration dict for each of the metrics"""
        self._metric_config_dict = {}

        if 'all' in self.config.metrics:
            metric_list = list_metrics(force_update=True)
        else:
            metric_list = self.config.metrics

        for metric_name_ in metric_list:
            if metric_name_ in self.config.exclude_metrics:
                continue
            sub_dict = self.config.metric_config.get('general', {}).copy()
            sub_dict.update(self.config.metric_config.get(metric_name_, {}))
            self._metric_config_dict[metric_name_] = sub_dict

    def _setup_iterator(self, handle_list):
        """Setup the iterator that runs in parallel over the handles"""

        itrs = [ self.input_iterator(tag) for tag in handle_list ]

        for it in zip(*itrs):
            data = []
            first = True
            for (s, e, d) in it:
                if first:
                    data.append(s)
                    data.append(e)
                    data.append(d)
                    first = False
                else:
                    data.append(d)
            yield data

    def _initialize_run(self):
        self._output_array_handle = None
        self._output_cache_handle = None

    def _do_chunk_output(self, cached_data, out_data, start, end, first):
        if first:
            self._output_array_handle = self.add_handle("output", data=out_data)
            self._output_array_handle.initialize_write(
                self._input_length, communicator=self.comm
            )
            self._output_cache_handle = self.add_handle("cache", data=cached_data)
            self._output_cache_handle.initialize_write(
                self._input_length, communicator=self.comm
            )
        self._output_array_handle.set_data(out_data, partial=True)
        self._output_array_handle.write_chunk(start, end)
        self._output_cache_handle.set_data(cached_data, partial=True)
        self._output_cache_handle.write_chunk(start, end)

    def _finalize_run(self):
        self._output_array_handle.finalize_write()
        self._output_cache_handle.finalize_write()
