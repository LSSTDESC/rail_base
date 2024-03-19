import numpy as np
from ceci.config import StageParameter as Param
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

        self._process_all_chunk_metrics(estimate_data, reference_data, start, end, first)


    def _process_all(self, data_tuple):
        estimate_data = data_tuple[0]
        reference_data = data_tuple[1]

        self._process_all_metrics(estimate_data, reference_data)
