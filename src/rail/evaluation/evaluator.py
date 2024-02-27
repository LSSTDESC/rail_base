"""
Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

import numpy as np

from ceci.config import StageParameter as Param
from ceci.stage import PipelineStage
from qp.metrics.pit import PIT
from rail.core.data import Hdf5Handle, QPHandle
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
    outputs = [("output", Hdf5Handle),
               ('summary', Hdf5Handle)]

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
        return dict(output=self.get_handle("output"), summary=self.get_handle("summary"))

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
        the_dict[subcls.metric_name] = subcls
    return the_dict
        

class BaseEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = 'BaseEvaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(
        metrics=Param(list, [], required=False,
            msg="The metrics you want to evaluate."),
        chunk_size=Param(int, 10000, required=False,
            msg="The default number of PDFs to evaluate per loop."),
        _random_state=Param(float, default=None, required=False,
            msg="Random seed value to use for reproducible results."),
    )

    outputs = [("output", Hdf5Handle),
               ('summary', Hdf5Handle)]

    metric_base_class = None
        
    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        self._output_handle = None
        self._summary_handle = None
        self._metric_dict = _build_metric_dict(self.metric_base_class)
        self._cached_data = {}
        self._cached_metrics = {}

    def run(self):
        print(f"Requested metrics: {self.config.metrics}")

        itr = self._setup_iterator()

        first = True
        for data_tuple in itr:
            chunk_start, chunk_end = data_tuple[0], data_tuple[1]

            print(f"Processing {self.rank} running evaluator on chunk {chunk_start} - {chunk_end}.")
            self._process_chunk(data_tuple, first)
            first = False

    def finalize(self):
        self._output_handle.finalize_write()
        summary_data = {}

        for metric, cached_metric in self._cached_metrics.items():
            if self.comm:
                self._cached_data[metric] = self.comm.gather(self._cached_data[metric])
            summary_data[metric] = np.array([cached_metric.finalize(self._cached_data[metric])])
        
        self._summary_handle = self.add_handle('summary', data=summary_data)
        PipelineStage.finalize(self)

    def _setup_iterator(self):
        """Setup the iterator that runs in parallel over the handles"""

        handle_list = [ input_[0] for input_ in self.inputs ]
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

    def _process_chunk(self, data_tuple, first):
        raise NotImplementedError('BaseEvaluator._process_chunk()')


    def _output_table_chunk_data(self, out_table, first):        
        out_table_to_write = {key: np.array(val).astype(float) for key, val in out_table.items()}

        if first:
            self._output_handle = self.add_handle('output', data=out_table_to_write)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(out_table_to_write, partial=True)
        self._output_handle.write_chunk(start, end)

    
