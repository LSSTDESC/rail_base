"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
import qp

from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.estimation.informer import PzInformer
from rail.core.data import QPHandle


class PointEstHistInformer(PzInformer):
    """Placeholder Informer"""

    name = "PointEstHistInformer"
    config_options = PzInformer.config_options.copy()

    def __init__(self, args, comm=None):
        PzInformer.__init__(self, args, comm=comm)

    def run(self):
        self.add_data("model", np.array([None]))


class PointEstHistSummarizer(PZSummarizer):
    """Summarizer which simply histograms a point estimate"""

    name = "PointEstHistSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        seed=Param(int, 87, msg="random seed"),
        point_estimate=Param(str, "zmode", msg="Which point estimate to use"),
        nsamples=Param(int, 1000, msg="Number of sample distributions to return"),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None
        self.bincents = None

    def run(self):
        iterator = self.input_iterator("input")
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        self.bincents = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])
        bootstrap_matrix = self._broadcast_bootstrap_matrix()
        # Initiallizing the histograms
        single_hist = np.zeros(self.config.nzbins)
        hist_vals = np.zeros((self.config.nsamples, self.config.nzbins))

        first = True
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            self._process_chunk(
                s, e, test_data, first, bootstrap_matrix, single_hist, hist_vals
            )
            first = False
        if self.comm is not None:  # pragma: no cover
            hist_vals, single_hist = self._join_histograms(hist_vals, single_hist)

        if self.rank == 0:
            sample_ens = qp.Ensemble(
                qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals))
            )
            qp_d = qp.Ensemble(
                qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(single_hist))
            )
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(
        self, start, end, test_data, _first, bootstrap_matrix, single_hist, hist_vals
    ):
        zb = test_data.ancil[self.config.point_estimate]
        single_hist += np.histogram(zb, bins=self.zgrid)[0]
        for i in range(self.config.nsamples):
            bootstrap_indeces = bootstrap_matrix[:, i]
            # Neither all of the bootstrap_draws are in this chunk nor the index starts at "start"
            mask = (bootstrap_indeces >= start) & (bootstrap_indeces < end)
            bootstrap_indeces = bootstrap_indeces[mask] - start
            zarr = zb[bootstrap_indeces]
            hist_vals[i] += np.histogram(zarr, bins=self.zgrid)[0]
