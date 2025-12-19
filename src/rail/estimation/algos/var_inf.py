"""
A summarizer that simple makes a histogram of a point estimate
"""

from typing import Any, Iterable

import numpy as np
import qp
from ceci.config import StageParameter as Param
from scipy.special import digamma
from scipy.stats import dirichlet

from rail.core.data import QPHandle
from rail.core.common_params import SharedParams
from rail.estimation.informer import PzInformer
from rail.estimation.summarizer import PZSummarizer

TEENY = 1.0e-15


class VarInfStackInformer(PzInformer):
    """Placeholder Informer"""

    name = "VarInfStackInformer"
    entrypoint_function = "inform"  # the user-facing science function for this class
    interactive_function = "var_inf_stack_informer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class VarInfStackSummarizer(PZSummarizer):
    """Variational inference summarizer based on notebook created by Markus Rau
    The summzarizer is appropriate for the likelihoods returned by
    template-based codes, for which the NaiveSummarizer are not appropriate.

    """

    name = "VarInfStackSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    interactive_function = "var_inf_stack_summarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=SharedParams.copy_param("zmin"),
        zmax=SharedParams.copy_param("zmax"),
        nzbins=SharedParams.copy_param("nzbins"),
        seed=Param(int, 87, msg="random seed"),
        n_iter=Param(
            int, 100, msg="The number of iterations in the variational inference"
        ),
        n_samples=Param(
            int, 500, msg="The number of samples used in dirichlet uncertainty"
        ),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.zgrid: np.ndarray | None = None

    def summarize(
        self, input_data: qp.Ensemble, **kwargs
    ) -> QPHandle | dict[str, QPHandle]:
        """Summarizer for VarInfStack which returns multiple items

        Parameters
        ----------
        input_data : qp.Ensemble
            Per-galaxy p(z), and any ancillary data associated with it

        Returns
        -------
        QPHandle | dict[str, QPHandle]
            Ensemble with n(z), and any ancillary data
            Return type depends on `output_mode`
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        if len(self.outputs) == 1 or self.config.output_mode != "return":
            results = self.get_handle("output")
        # if there is more than one output and output_mode = return, return them all as a dictionary
        elif len(self.outputs) > 1 and self.config.output_mode == "return":
            results = {}
            for output in self.outputs:
                results[output[0]] = self.get_handle(output[0])
        return results

    def _setup_iterator(self) -> Iterable:
        input_handle = self.get_handle("input", allow_missing=True)
        try:
            self.config.hdf5_groupname
        except Exception:
            self.config.hdf5_groupname = None
        input_length = input_handle.size(groupname=self.config.hdf5_groupname)
        chunk_size = int(np.ceil(input_length / self.size))

        iterator = self.input_iterator("input", chunk_size=chunk_size)
        return iterator

    def run(self) -> None:
        # Redefining the chunk size so that all of the data is distributed at once in the
        # nodes. This would fill all the memory if not enough nodes are allocated

        iterator = self._setup_iterator()
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
        first = True
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            alpha_trace = self._process_chunk(s, e, test_data, first)
            first = False

        if self.rank == 0:
            # old way of just spitting out a single distribution
            # qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace))
            # instead, sample and save the samples
            rng = np.random.default_rng(seed=self.config.seed)
            sample_pz = dirichlet.rvs(
                alpha_trace, size=self.config.n_samples, random_state=rng
            )
            qp_d = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=alpha_trace)
            )

            sample_ens = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=sample_pz)
            )
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(
        self, _start: int, _end: int, test_data: qp.Ensemble, first: bool
    ) -> np.ndarray:
        if not first:  # pragma: no cover
            raise ValueError(
                "This algorithm needs all data in memory at once, increase nprocess or chunk size."
            )

        # Initiallizing arrays
        assert self.zgrid is not None
        alpha_trace = np.ones(len(self.zgrid))
        init_trace = np.ones(len(self.zgrid))
        pdf_vals = test_data.pdf(self.zgrid)
        log_pdf_vals = np.log(np.array(pdf_vals) + TEENY)
        for _ in range(self.config.n_iter):
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
