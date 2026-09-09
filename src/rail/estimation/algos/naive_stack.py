"""
A summarizer that simple makes a histogram of a point estimate
"""

from typing import Any

import numpy as np
import qp
from ceci.config import StageParameter as Param
import gc

from rail.core.data import QPHandle, TableHandle, TableLike
from rail.core.common_params import SharedParams, TOMOGRAPHY_ALL, TOMOGRAPHY_NONE
from rail.estimation.informer import PzInformer
from rail.estimation.summarizer import PZSummarizer


class NaiveStackInformer(PzInformer):
    """Placeholder Informer"""

    name = "NaiveStackInformer"
    entrypoint_function = "inform"  # the user-facing science function for this class
    interactive_function = "naive_stack_informer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class NaiveStackSummarizer(PZSummarizer):
    """Summarizer which stacks individual P(z)"""

    name = "NaiveStackSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    interactive_function = "naive_stack_summarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=SharedParams.copy_param("zmin"),
        zmax=SharedParams.copy_param("zmax"),
        nzbins=SharedParams.copy_param("nzbins"),
        seed=Param(int, 87, msg="random seed"),
        n_samples=Param(int, 1000, msg="Number of sample distributions to create"),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.zgrid: np.ndarray | None = None


    def run(self) -> None:
        handle = self.get_handle("input", allow_missing=True)
        self._input_length = handle.size()
        iterator = self._setup_iterator()
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        assert self.zgrid is not None
        # Initializing the stacking pdf's
        n_tomo_bins = self._get_n_tomo_bins()

        n_objects = np.zeros((n_tomo_bins), dtype=int)
        yvals = np.zeros((n_tomo_bins, len(self.zgrid)))
        bvals = np.zeros((n_tomo_bins, self.config.n_samples, len(self.zgrid)))

        first = True
        for s, e, test_data, mask in iterator:
            self.log.info(f"Process {self.rank} running summarizer on chunk {s:,} - {e:,}")
            self._process_chunk(
                s, e, test_data, mask, first, yvals, bvals, n_objects
            )
            gc.collect()
            first = False
        if self.comm is not None:  # pragma: no cover
            bvals, yvals, n_objects = self._join_histograms(bvals, yvals, n_objects)

        if self.rank == 0:
            sample_ens = qp.Ensemble(
                qp.interp, data=dict(xvals=self.zgrid, yvals=np.vstack(bvals))
            )
            qp_d = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=yvals))
            i_realization=np.arange(self.config.n_samples)
            if n_tomo_bins > 1:
                bin_idx = np.arange(self.config.selected_bin, n_tomo_bins)
            elif 'selected_bin' in self.config:
                bin_idx = [self.config.selected_bin]
            else:
                bin_idx = [TOMOGRAPHY_ALL]
            sample_ens.set_ancil(
                dict(
                    bin_idx=np.repeat(bin_idx, self.config.n_samples),
                    i_realization=np.tile(i_realization, n_tomo_bins),
                )
            )
            qp_d.set_ancil(
                dict(
                    bin_idx=np.squeeze(np.array(bin_idx)),
                    n_objects=np.squeeze(np.array(n_objects)),
                )
            )
            self.add_data("output", sample_ens)
            self.add_data("single_NZ", qp_d)

    def _process_chunk(
        self,
        start: int,
        end: int,
        data: qp.Ensemble,
        mask: np.ndarray,
        _first: bool,
        yvals: np.ndarray,
        bvals: np.ndarray,
        n_objects: np.ndarray,
    ) -> None:
        assert self.zgrid is not None
        pdf_vals = data.pdf(self.zgrid)
        squeeze_mask = np.squeeze(mask)

        n_dim = len(squeeze_mask.shape)
        if n_dim == 1:
            masks = [squeeze_mask]
        else:
            masks = squeeze_mask

        for i, mask_ in enumerate(masks):
            n_objects[i] += mask_.sum()
            yvals[i] += np.sum(
                np.where(
                    np.isfinite(pdf_vals[mask_, :]), pdf_vals[mask_], 0.0
                ),
                axis=0,
            )
            # qp_d is the normalized probability of the stack, we need to know how many galaxies were
            rng = np.random.default_rng(seed=[self.config.seed, start])
            for j in range(self.config.n_samples):
                # This is Poisson bootstrap, a variant of regular bootstrap
                # that does not require anything to be stored or comunicated between
                # processes. For large numbers of objects this converges to the same
                # distribution as regular bootstrap.
                bootstrap_weights = rng.poisson(lam=1.0, size=pdf_vals.shape[0])
                bvals[i][j] += bootstrap_weights[mask_] @ pdf_vals[mask_]


class NaiveStackMaskedSummarizer(NaiveStackSummarizer):
    name = "NaiveStackMaskedSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    interactive_function = "naive_stack_masked_summarizer"
    config_options = NaiveStackSummarizer.config_options.copy()
    config_options.update(
        selected_bin=Param(int, TOMOGRAPHY_NONE, msg=f"bin to use, or {TOMOGRAPHY_ALL} for all bins >=0 or {TOMOGRAPHY_NONE} for no masking"),
        n_tomo_bins=Param(int, 1, msg="Number of tomographic bins"),
    )
    inputs = [("input", QPHandle), ("tomography_bins", TableHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def summarize(
        self, input_data: qp.Ensemble, tomo_bins: TableLike | None = None, **kwargs
    ) -> QPHandle:
        """Override the Summarizer.summarize() method to take tomo bins
        as an additional input

        Parameters
        ----------
        input_data : qp.Ensemble
            Per-galaxy p(z), and any ancillary data associated with it

        tomo_bins : TableLike | None, optional
            Tomographic bins file, by default None

        Returns
        -------
        QPHandle
            Ensemble with n(z), and any ancillary data
        """
        self.set_data("input", input_data)
        if tomo_bins is None:
            self.config.tomography_bins = None
        else:
            self.set_data("tomography_bins", tomo_bins)
        self.run()
        self.finalize()
        return self._do_return()
