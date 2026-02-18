"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
import qp

from rail.estimation.estimator import PzEstimator
from rail.estimation.informer import PzInformer


class GaussianPzInformer(PzInformer):
    """Placeholder Informer"""

    name = "GaussianPzInformer"
    entrypoint_function = "inform"  # the user-facing science function for this class
    interactive_function = "gaussian_pz_informer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class GaussianPzEstimator(PzEstimator):
    """Estimator which converts to Gaussian reps"""

    name = "GaussianPzEstimator"
    entrypoint_function = "estimate"  # the user-facing science function for this class
    interactive_function = "gaussian_pz_estimator"
    config_options = PzEstimator.config_options.copy()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:

        mean = np.squeeze(data.mean())
        std = np.squeeze(data.std())
        std = np.where(np.isfinite(std), std, 0.01)

        qp_dstn = qp.Ensemble(qp.stats.norm, data=dict(loc=mean, scale=std))
        self._do_chunk_output(qp_dstn, start, end, first)
