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


class PointEstHistInformer(PzInformer):
    """Placeholder Informer"""

    name = "PointEstHistInformer"
    entrypoint_function = "inform"  # the user-facing science function for this class
    interactive_function = "point_est_hist_informer"
    config_options = PzInformer.config_options.copy()

    def _finalize_run(self) -> None:
        self.model = np.array([None])
        PzInformer._finalize_run(self)


class PointEstHistSummarizer(PZSummarizer):
    """Summarizer which simply histograms a point estimate"""

    name = "PointEstHistSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    interactive_function = "point_est_hist_summarizer"
    config_options = PZSummarizer.config_options.copy()
    config_options.update(
        zmin=SharedParams.copy_param("zmin"),
        zmax=SharedParams.copy_param("zmax"),
        nzbins=SharedParams.copy_param("nzbins"),
        seed=Param(int, 87, msg="random seed"),
        point_estimate_key=Param(str, "zmode", msg="Which point estimate to use"),
        n_samples=Param(int, 1000, msg="Number of sample distributions to return"),
    )
    inputs = [("input", QPHandle)]
    outputs = [("output", QPHandle), ("single_NZ", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.zgrid: np.ndarray | None = None
        self.bincents: np.ndarray | None = None

    def run(self) -> None:
        handle = self.get_handle("input", allow_missing=True)
        self._input_length = handle.size()
        iterator = self._setup_iterator()
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        assert self.zgrid is not None

        self.bincents = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])
        # Initiallizing the histograms
        n_tomo_bins = self._get_n_tomo_bins()
        n_objects = np.zeros((n_tomo_bins), dtype=int)
        single_hist = np.zeros((n_tomo_bins, self.config.nzbins))
        hist_vals = np.zeros((n_tomo_bins, self.config.n_samples, self.config.nzbins))

        first = True
        for s, e, test_data, mask in iterator:
            self.log.info(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(
                s, e, test_data, mask, first, single_hist, hist_vals, n_objects
            )
            first = False
            gc.collect()
            del test_data
        if self.comm is not None:  # pragma: no cover
            single_hist, hist_vals, n_objects = self._join_histograms(single_hist, hist_vals, n_objects)

        if self.rank == 0:
            sample_ens = qp.Ensemble(
                qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(hist_vals))
            )
            qp_d = qp.Ensemble(
                qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(single_hist))
            )
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
        test_data: qp.Ensemble,
        mask: np.ndarray,
        _first: bool,
        single_hist: np.ndarray,
        hist_vals: np.ndarray,
        n_objects: np.ndarray,
    ) -> None:
        assert self.zgrid is not None
        zb = test_data.ancil[self.config.point_estimate_key]
        squeeze_mask = np.squeeze(mask)

        n_dim = len(squeeze_mask.shape)
        if n_dim == 1:
            masks = [squeeze_mask]
        else:
            masks = squeeze_mask

        for i, mask_ in enumerate(masks):
            n_objects[i] += mask_.sum()
            single_hist[i] += np.histogram(zb[mask_], bins=self.zgrid)[0]
            # get a new random seed for each chunk, but make it deterministic
            # by using the chunk start index and the base seed together
            rng = np.random.default_rng(seed=[self.config.seed, start])
            for j in range(self.config.n_samples):
                # poisson bootstrap - see naive_stack.py comment for details.
                bootstrap_weights = rng.poisson(lam=1.0, size=zb.size)
                hist_vals[i][j] += np.histogram(zb, weights=bootstrap_weights, bins=self.zgrid)[0]


class PointEstHistMaskedSummarizer(PointEstHistSummarizer):
    """Summarizer which simply histograms a point estimate"""

    name = "PointEstHistMaskedSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    interactive_function = "point_est_hist_masked_summarizer"
    config_options = PointEstHistSummarizer.config_options.copy()
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
            Per-galaxy p(z), and any ancilary data associated with it

        tomo_bins : TableLike | None, optional
            Tomographic bins file, by default None

        Returns
        -------
        QPHandle
            Ensemble with n(z), and any ancilary data
        """
        self.set_data("input", input_data)
        if tomo_bins is None:
            self.config.tomography_bins = None
        else:
            self.set_data("tomography_bins", tomo_bins)
        self.run()
        self.finalize()
        return self._do_return()
