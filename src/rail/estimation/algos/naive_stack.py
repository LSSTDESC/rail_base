"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
import qp

from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.estimation.informer import PzInformer
from rail.core.data import QPHandle


class NaiveStackInformer(PzInformer):
    """Placeholder Informer"""

    name = "NaiveStackInformer"
    config_options = PzInformer.config_options.copy()

    def __init__(self, args, comm=None):
        PzInformer.__init__(self, args, comm=comm)

    def run(self):
        self.add_data("model", np.array([None]))


class NaiveStackSummarizer(PZSummarizer):
    """Summarizer which stacks individual P(z)"""

    name = "NaiveStackSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        seed=Param(int, 87, msg="random seed"),
        nsamples=Param(int, 1000, msg="Number of sample distributions to create"),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        iterator = self.input_iterator("input")
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        # Initiallizing the stacking pdf's
        yvals = np.zeros((1, len(self.zgrid)))
        bvals = np.zeros((self.config.nsamples, len(self.zgrid)))
        bootstrap_matrix = self._broadcast_bootstrap_matrix()

        first = True
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            self._process_chunk(s, e, test_data, first, bootstrap_matrix, yvals, bvals)
            first = False
        if self.comm is not None:  # pragma: no cover
            bvals, yvals = self._join_histograms(bvals, yvals)

        if self.rank == 0:
            sample_ens = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=bvals)
            )
            qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=yvals))
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(self, start, end, data, _first, bootstrap_matrix, yvals, bvals):
        pdf_vals = data.pdf(self.zgrid)
        yvals += np.expand_dims(
            np.sum(np.where(np.isfinite(pdf_vals), pdf_vals, 0.0), axis=0), 0
        )
        # qp_d is the normalized probability of the stack, we need to know how many galaxies were
        for i in range(self.config.nsamples):
            bootstrap_draws = bootstrap_matrix[:, i]
            # Neither all of the bootstrap_draws are in this chunk nor the index starts at "start"
            mask = (bootstrap_draws >= start) & (bootstrap_draws < end)
            bootstrap_draws = bootstrap_draws[mask] - start
            bvals[i] += np.sum(pdf_vals[bootstrap_draws], axis=0)
