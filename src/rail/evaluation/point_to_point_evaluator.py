import numpy as np
from ceci.config import StageParameter as Param
from qp.metrics.point_estimate_metric_classes import PointToPointMetric

from rail.core.data import TableHandle, QPHandle
from rail.evaluation.evaluator import Evaluator


class PointToPointEvaluator(Evaluator):
    """Evaluate the performance of a photo-z estimator against reference point estimate"""

    name = "PointToPointEvaluator"
    config_options = Evaluator.config_options.copy()
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
        estimate_data = np.squeeze(data_tuple[2].ancil[self.config.point_estimate_key])
        reference_data = data_tuple[3][self.config.reference_dictionary_key]

        self._process_all_chunk_metrics(estimate_data, reference_data, start, end, first)


    def _process_all(self, data_tuple):
        estimate_data = np.squeeze(data_tuple[0].ancil[self.config.point_estimate_key])
        reference_data = data_tuple[1][self.config.hdf5_groupname][self.config.reference_dictionary_key]

        self._process_all_metrics(estimate_data, reference_data)
