"""
A summarizer that simple makes a histogram of a point estimate
"""

import numpy as np
import qp
from scipy.special import digamma
from scipy.stats import dirichlet
from ceci.config import StageParameter as Param
from rail.estimation.summarizer import PZSummarizer
from rail.estimation.informer import PzInformer
from rail.core.data import QPHandle

TEENY = 1.0e-15


class VarInfStackInformer(PzInformer):
    """Placeholder Informer"""

    name = "VarInfStackInformer"
    config_options = PzInformer.config_options.copy()

    def __init__(self, args, comm=None):
        PzInformer.__init__(self, args, comm=comm)

    def run(self):
        self.add_data("model", np.array([None]))


class VarInfStackSummarizer(PZSummarizer):
    """Variational inference summarizer based on notebook created by Markus Rau
    The summzarizer is appropriate for the likelihoods returned by
    template-based codes, for which the NaiveSummarizer are not appropriate.

    Parameters
    ----------
    zmin: float
      minimum z for redshift grid
    zmax: float
      maximum z for redshift grid
    nzbins: int
      number of bins for redshift grid
    niter: int
      number of iterations to perform in the variational inference
    nsamples: int
      number of samples used in dirichlet to determind error bar
    """

    name = "VarInfStackSummarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
        zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
        nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
        seed=Param(int, 87, msg="random seed"),
        niter=Param(
            int, 100, msg="The number of iterations in the variational inference"
        ),
        nsamples=Param(
            int, 500, msg="The number of samples used in dirichlet uncertainty"
        ),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args, comm=None):
        PZSummarizer.__init__(self, args, comm=comm)
        self.zgrid = None

    def run(self):
        # Redefining the chunk size so that all of the data is distributed at once in the
        # nodes. This would fill all the memory if not enough nodes are allocated

        input_data = self.get_handle("input", allow_missing=True)
        try:
            self.config.hdf5_groupname
        except Exception:
            self.config.hdf5_groupname = None
        input_length = input_data.size(groupname=self.config.hdf5_groupname)
        self.config.chunk_size = np.ceil(input_length / self.size)

        iterator = self.input_iterator("input")
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        first = True
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            alpha_trace = self._process_chunk(s, e, test_data, first)
            first = False

        if self.rank == 0:
            # old way of just spitting out a single distribution
            # qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace))
            # instead, sample and save the samples
            rng = np.random.default_rng(seed=self.config.seed)
            sample_pz = dirichlet.rvs(
                alpha_trace, size=self.config.nsamples, random_state=rng
            )
            qp_d = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace)
            )

            sample_ens = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=sample_pz)
            )
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(self, _start, _end, test_data, first):
        if not first:  # pragma: no cover
            raise ValueError(
                "This algorithm needs all data in memory at once, increase nprocess or chunk size."
            )

        # Initiallizing arrays
        alpha_trace = np.ones(len(self.zgrid))
        init_trace = np.ones(len(self.zgrid))
        pdf_vals = test_data.pdf(self.zgrid)
        log_pdf_vals = np.log(np.array(pdf_vals) + TEENY)
        for _ in range(self.config.niter):
            dig = np.array(
                [digamma(kk) - digamma(np.sum(alpha_trace)) for kk in alpha_trace]
            )
            matrix_grid = np.exp(dig + log_pdf_vals)
            gamma_matrix = np.array([kk / np.sum(kk) for kk in matrix_grid])
            for kk in matrix_grid:
                break
            nk_partial = np.sum(gamma_matrix, axis=0)
            if self.comm is not None:  # pragma: no cover
                nk = self.comm.allreduce(nk_partial)
            else:
                nk = nk_partial
            alpha_trace = nk + init_trace
        return alpha_trace
