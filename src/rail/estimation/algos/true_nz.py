"""
A summarizer-like stage that simple makes a histogram the true nz
"""

import numpy as np
import qp
from ceci.config import StageParameter as Param

from rail.core.common_params import SHARED_PARAMS
from rail.core.data import QPHandle, TableHandle
from rail.core.stage import RailStage


class TrueNZHistogrammer(RailStage):
    """Summarizer-like stage which simply histograms the true redshift"""

    name = "TrueNZHistogrammer"
    config_options = RailStage.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        selected_bin=Param(int, -1, msg="Which tomography bin to consider"),
        chunk_size=SHARED_PARAMS,
        hdf5_groupname=SHARED_PARAMS,
    )
    inputs = [("input", TableHandle), ("tomography_bins", TableHandle)]
    outputs = [("true_NZ", QPHandle)]

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.zgrid = None
        self.bincents = None

    def _setup_iterator(self):
        itrs = [
            self.input_iterator("input", groupname=self.config.hdf5_groupname),
            self.input_iterator("tomography_bins", groupname=""),
        ]

        for it in zip(*itrs):
            first = True
            mask = None
            for s, e, d in it:
                if first:
                    start = s
                    end = e
                    pz_data = d
                    first = False
                else:
                    if self.config.selected_bin < 0:
                        mask = np.ones(e - s, dtype=bool)
                    else:
                        mask = d["class_id"] == self.config.selected_bin
            yield start, end, pz_data, mask

    def run(self):
        iterator = self._setup_iterator()
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        self.bincents = 0.5 * (self.zgrid[1:] + self.zgrid[:-1])
        # Initiallizing the histograms
        single_hist = np.zeros(self.config.nzbins)

        first = True
        for s, e, data, mask in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            self._process_chunk(s, e, data, mask, first, single_hist)
            first = False
        if self.comm is not None:  # pragma: no cover
            single_hist = self.comm.reduce(single_hist)

        if self.rank == 0:
            n_total = single_hist.sum()
            qp_d = qp.Ensemble(
                qp.hist,
                data=dict(bins=self.zgrid, pdfs=np.atleast_2d(single_hist)),
                ancil=dict(n_total=np.array([n_total], dtype=int)),
            )
            self.add_data("true_NZ", qp_d)

    def _process_chunk(
        self,
        _start,
        _end,
        data,
        mask,
        _first,
        single_hist,
    ):
        squeeze_mask = np.squeeze(mask)
        zb = data[self.config.redshift_col][squeeze_mask]
        single_hist += np.histogram(zb, bins=self.zgrid)[0]

    def histogram(self, catalog, tomo_bins):
        """The main interface method for ``TrueNZHistogrammer``.

        Creates histogram of N of Z_true.

        This will attach the sample to this `Stage` (for introspection and
        provenance tracking).

        Then it will call the run() and finalize() methods, which need to be
        implemented by the sub-classes.

        The run() method will need to register the data that it creates to this
        Estimator by using ``self.add_data('output', output_data)``.

        Finally, this will return a PqHandle providing access to that output
        data.

        Parameters
        ----------
        catalog : table-like
            The sample with the true NZ column

        Returns
        -------
        output_data : QPHandle
            A handle giving access to a the histogram in QP format
        """
        self.set_data("input", catalog)
        self.set_data("tomography_bins", tomo_bins)
        self.run()
        self.finalize()
        return self.get_handle("true_NZ")
