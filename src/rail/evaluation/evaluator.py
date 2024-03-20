"""
Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

import numpy as np
import qp
from ceci.config import StageParameter as Param
from ceci.stage import PipelineStage
from qp.metrics.pit import PIT
from qp.metrics.base_metric_classes import MetricOutputType
from rail.core.data import Hdf5Handle, QPHandle, QPDictHandle
from rail.core.stage import RailStage
from rail.core.common_params import SHARED_PARAMS
from rail.evaluation.metrics.cdeloss import CDELoss
from rail.evaluation.metrics.pointestimates import (
    PointSigmaIQR,
    PointBias,
    PointOutlierRate,
    PointSigmaMAD,
)


class Evaluator(RailStage):
    """Evaluate the performance of a photo-Z estimator"""

    name = "Evaluator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        zmin=Param(float, 0., msg="min z for grid"),
        zmax=Param(float, 3.0, msg="max z for grid"),
        nzbins=Param(int, 301, msg="# of bins in zgrid"),
        pit_metrics=Param(str, 'all', msg='PIT-based metrics to include'),
        point_metrics=Param(str, 'all', msg='Point-estimate metrics to include'),
        hdf5_groupname=Param(str, '', msg='Name of group in hdf5 where redshift data is located'),
        do_cde=Param(bool, True, msg='Evaluate CDE Metric'),
        redshift_col=SHARED_PARAMS,
    )
    inputs = [('input', QPHandle),
              ('truth', Hdf5Handle)]
    outputs = [("output", Hdf5Handle)]

    def __init__(self, args, comm=None):
        """Initialize Evaluator"""
        RailStage.__init__(self, args, comm=comm)

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

        self.set_data("input", data)
        self.set_data("truth", truth)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def run(self):
        """Run method

        Evaluate all the metrics and put them into a table

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Get the truth data from the data store under this stages 'truth' tag
        Puts the data into the data store under this stages 'output' tag
        """

        pz_data = self.get_data('input')
        if self.config.hdf5_groupname:  # pragma: no cover
            specz_data = self.get_data('truth')[self.config.hdf5_groupname]
        else: 
            specz_data = self.get_data('truth')
        z_true = specz_data[self.config['redshift_col']]

        zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)

        # Create an instance of the PIT class
        pitobj = PIT(pz_data, z_true)

        # Build reference dictionary of the PIT meta-metrics from this PIT instance
        PIT_METRICS = dict(
            AD=getattr(pitobj, "evaluate_PIT_anderson_ksamp"),
            CvM=getattr(pitobj, "evaluate_PIT_CvM"),
            KS=getattr(pitobj, "evaluate_PIT_KS"),
            OutRate=getattr(pitobj, "evaluate_PIT_outlier_rate"),
        )

        # Parse the input configuration to determine which meta-metrics should be calculated
        if self.config.pit_metrics == "all":
            pit_metrics = list(PIT_METRICS.keys())
        else:  # pragma: no cover
            pit_metrics = self.config.pit_metrics.split()

        # Evaluate each of the requested meta-metrics, and store the result in `out_table`
        out_table = {}
        for pit_metric in pit_metrics:
            value = PIT_METRICS[pit_metric]()

            # The result objects of some meta-metrics are bespoke scipy objects with inconsistent fields.
            # Here we do our best to store the relevant fields in `out_table`.
            if isinstance(value, list):  # pragma: no cover
                out_table[f"PIT_{pit_metric}"] = value
            else:
                out_table[f"PIT_{pit_metric}_stat"] = [
                    getattr(value, "statistic", None)
                ]
                out_table[f"PIT_{pit_metric}_pval"] = [getattr(value, "p_value", None)]
                out_table[f"PIT_{pit_metric}_significance_level"] = [
                    getattr(value, "significance_level", None)
                ]

        POINT_METRICS = dict(
            SimgaIQR=PointSigmaIQR,
            Bias=PointBias,
            OutlierRate=PointOutlierRate,
            SigmaMAD=PointSigmaMAD,
        )
        if self.config.point_metrics == "all":
            point_metrics = list(POINT_METRICS.keys())
        else:  # pragma: no cover
            point_metrics = self.config.point_metrics.split()

        z_mode = None
        for point_metric in point_metrics:
            if z_mode is None:
                z_mode = np.squeeze(pz_data.mode(grid=zgrid))
            value = POINT_METRICS[point_metric](z_mode, z_true).evaluate()
            out_table[f"POINT_{point_metric}"] = [value]

        if self.config.do_cde:
            value = CDELoss(pz_data, zgrid, z_true).evaluate()
            out_table["CDE_stat"] = [value.statistic]
            out_table["CDE_pval"] = [value.p_value]

        # Converting any possible None to NaN to write it
        out_table_to_write = {
            key: np.array(val).astype(float) for key, val in out_table.items()
        }
        self.add_data("output", out_table_to_write)


def _all_subclasses(a_class):
    return set(a_class.__subclasses__()).union(
        [s for c in a_class.__subclasses__() for s in _all_subclasses(c)]
    )


def _build_metric_dict(a_class):
    the_dict = {}
    for subcls in _all_subclasses(a_class):
        if subcls.metric_name is None:
            continue
        the_dict[subcls.metric_name] = subcls
    return the_dict


class BaseEvaluator(RailStage):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "BaseEvaluator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(
            list, [], required=False, msg="The metrics you want to evaluate."
        ),
        exclude_metrics=Param(
            list, [], msg="List of metrics to exclude", required=False
        ),
        metric_config=Param(
            dict, msg="configuration of individual_metrics", default={}
        ),
        chunk_size=Param(
            int,
            10000,
            required=False,
            msg="The default number of PDFs to evaluate per loop.",
        ),
        _random_state=Param(
            float,
            default=None,
            required=False,
            msg="Random seed value to use for reproducible results.",
        ),
        force_exact=Param(
            bool,
            default=False,
            required=False,
            msg="Force the exact calculation.  This will not allow parallelization",
        ),
    )

    outputs = [("output", Hdf5Handle),
               ('summary', Hdf5Handle),
               ('single_distribution_summary', QPDictHandle),
            ]

    metric_base_class = None

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        self._output_handle = None
        self._summary_handle = None
        self._single_distribution_summary_handle = None
        self._metric_dict = _build_metric_dict(self.metric_base_class)
        self._cached_data = {}
        self._cached_metrics = {}
        self._single_distribution_summary_data = {}
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

        self.set_data("input", data)
        self.set_data("truth", truth)
        self.run()
        self.finalize()
        return {
            'output':self.get_handle("output"),
            'summary':self.get_handle("summary"),
            'single_distribution_summary':self.get_handle("single_distribution_summary"),
        }

    def run(self):
        self._build_config_dict()

        print(f"Requested metrics: {list(self._metric_config_dict.keys())}")

        if self.config.force_exact:
            self.run_single_node()
            return

        itr = self._setup_iterator()

        first = True
        for data_tuple in itr:
            chunk_start, chunk_end = data_tuple[0], data_tuple[1]

            print(
                f"Processing {self.rank} running evaluator on chunk {chunk_start} - {chunk_end}."
            )
            self._process_chunk(data_tuple, first)
            first = False

    def run_single_node(self):
        data_tuple = self._get_all_data()
        self._process_all(data_tuple)

    def finalize(self):
        if not self.config.force_exact:

            # Finalize all the metrics that produce a single value summary
            summary_data = {}
            for metric, cached_metric in self._cached_metrics.items():
                if cached_metric.metric_output_type != MetricOutputType.single_value:
                    continue
                if metric not in self._cached_data:
                    print(f"Skipping {metric} which did not cache data")
                    continue
                if self.comm:  # pragma: no cover
                    self._cached_data[metric] = self.comm.gather(self._cached_data[metric])
                summary_data[metric] = np.array([cached_metric.finalize(self._cached_data[metric])])

            self._summary_handle = self.add_handle('summary', data=summary_data)

            # Finalize all the metrics that produce a single distribution summaries
            single_distribution_summary_data = {}
            for metric, cached_metric in self._cached_metrics.items():
                if cached_metric.metric_output_type != MetricOutputType.single_distribution:
                    continue
                if metric not in self._cached_data:  # pragma: no cover
                    print(f"Skipping {metric} which did not cache data")
                    continue
                if self.comm:  # pragma: no cover
                    self._cached_data[metric] = self.comm.gather(self._cached_data[metric])

                # we expected `cached_metric.finalize` to return a qp.Ensemble
                single_distribution_summary_data[cached_metric.metric_name] = cached_metric.finalize([self._cached_data[metric]])

            self._single_distribution_summary_handle = self.add_handle('single_distribution_summary', data=single_distribution_summary_data)

        PipelineStage.finalize(self)

    def _setup_iterator(self, itrs=None):
        """Setup the iterator that runs in parallel over the handles"""

        if itrs is None:
            handle_list = [input_[0] for input_ in self.inputs]
            itrs = [self.input_iterator(tag) for tag in handle_list]

        for it in zip(*itrs):
            data = []
            first = True
            for s, e, d in it:
                if first:
                    data.append(s)
                    data.append(e)
                    data.append(d)
                    first = False
                else:
                    data.append(d)
            yield data

    def _get_all_data(self):
        """Stuff the data from all the handles into a tuple"""
        handles = [input_[0] for input_ in self.inputs]
        all_data = [self.get_data(handle_) for handle_ in handles]
        return all_data

    def _process_chunk(self, data_tuple, first):  # pragma: no cover
        raise NotImplementedError("BaseEvaluator._process_chunk()")

    def _process_all_chunk_metrics(self, estimate_data, reference_data, start, end, first):
        """This function takes the properly formatted data and loops over all the
        requested metrics. The metric outputs are either cached for later finalization
        or written to output files.

        Parameters
        ----------
        estimate_data : Ensemble or array
            The estimated values (of the appropriate type, float or pdf) to be used
            by the requested metrics.
        reference_data : Ensemble or array
            The reference or known values (of the appropriate type, float or pdf)
            to be used by the requested metrics.
        start : int
            The start index of the data chunk, used to write metric results to
            the correct location of the output file.
        end : int
            The end index of the data chunk, used to write metric results to the
            correct location of the output file.
        first : bool
            Used internally to determine if the output file should be initialized.

        Raises
        ------
        ValueError
            Raises an error if an unknown metric is requested.
        """
        out_table = {}
        for metric, this_metric in self._cached_metrics.items():
            if metric not in self._metric_dict:  # pragma: no cover
                raise ValueError(
                    f"Unsupported metric requested: '{metric}'. Available metrics are: {sorted(self._metric_dict.keys())}"
                )

            if this_metric.metric_output_type == MetricOutputType.single_value:
                if not hasattr(this_metric, "accumulate"):  # pragma: no cover
                    print(
                        f"{metric} with output type single_value does not support parallel processing yet"
                    )
                    continue

                centroids = this_metric.accumulate(estimate_data, reference_data)
                if self.comm:  # pragma: no cover
                    self._cached_data[metric] = centroids
                else:
                    if metric in self._cached_data:
                        self._cached_data[metric].append(centroids)
                    else:
                        self._cached_data[metric] = [centroids]

            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                if not hasattr(this_metric, 'accumulate'):  # pragma: no cover
                    print(f"{metric} with output type MetricOutputType.single_value does not support parallel processing yet")
                    continue
                self._cached_data[metric] = this_metric.accumulate(estimate_data, reference_data)
            else:
                out_table[metric] = this_metric.evaluate(estimate_data, reference_data,)

        self._output_table_chunk_data(start, end, out_table, first)


    def _output_table_chunk_data(self, start, end, out_table, first):
        out_table_to_write = {
            key: np.array(val).astype(float) for key, val in out_table.items()
        }

        if first:
            self._output_handle = self.add_handle("output", data=out_table_to_write)
            self._output_handle.initialize_write(
                self._input_length, communicator=self.comm
            )
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)


    def _process_all(self, data_tuple):  # pragma: no cover
        raise NotImplementedError("BaseEvaluator._process_all()")


    def _process_all_metrics(self, estimate_data, reference_data):
        """This function writes out metric values when operating in non-parallel mode.

        Parameters
        ----------
        estimate_data : Ensemble or array
            The estimated values (of the appropriate type, float or pdf) to be used
            by the requested metrics.
        reference_data : Ensemble or array
            The reference or known values (of the appropriate type, float or pdf)
            to be used by the requested metrics.

        Raises
        ------
        ValueError
            Raises an error if an unknown metric is requested.
        """

        out_table = {}
        summary_table = {}
        single_distribution_summary = {}

        for metric, this_metric in self._cached_metrics.items():
            if metric not in self._metric_dict:  # pragma: no cover
                raise ValueError(
                    f"Unsupported metric requested: '{metric}'. Available metrics are: {sorted(self._metric_dict.keys())}"
                )

            metric_result = this_metric.evaluate(estimate_data, reference_data)

            if this_metric.metric_output_type == MetricOutputType.single_value:
                summary_table[metric] = np.array([metric_result])
            elif this_metric.metric_output_type == MetricOutputType.single_distribution:
                single_distribution_summary[this_metric.metric_name] = metric_result
            else:
                out_table[metric] = metric_result

        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}
        self._output_handle = self.add_handle('output', data=out_table_to_write)
        self._summary_handle = self.add_handle('summary', data=summary_table)
        self._single_distribution_summary_handle = self.add_handle('single_distribution_summary', data=single_distribution_summary)


    def _build_config_dict(self):
        """Build the configuration dict for each of the metrics"""
        self._metric_config_dict = {}

        if "all" in self.config.metrics:  # pragma: no cover
            metric_list = list(self._metric_dict.keys())
        else:
            metric_list = self.config.metrics

        for metric_name_ in metric_list:
            if metric_name_ in self.config.exclude_metrics:  # pragma: no cover
                continue
            if metric_name_ not in self._metric_dict:
                print(
                    f"Unsupported metric requested: '{metric_name_}'.  "
                    f"Available metrics are: {sorted(self._metric_dict.keys())}"
                )
                continue

            sub_dict = self.config.metric_config.get("general", {}).copy()
            sub_dict.update(self.config.metric_config.get(metric_name_, {}))
            self._metric_config_dict[metric_name_] = sub_dict
            this_metric_class = self._metric_dict[metric_name_]
            this_metric = this_metric_class(**sub_dict)
            self._cached_metrics[metric_name_] = this_metric
