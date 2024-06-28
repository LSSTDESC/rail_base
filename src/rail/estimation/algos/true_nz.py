"""
A summarizer-like stage that simple makes a histogram the true nz
"""

import numpy as np
import qp

from ceci.config import StageParameter as Param
from rail.core.common_params import SHARED_PARAMS
from rail.core.stage import RailStage
from rail.core.data import QPHandle, TableHandle


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
        chunk_size=10000,
        hdf5_groupname="",
    )
    inputs = [("input", TableHandle), ("tomography_bins", TableHandle)]
    outputs = [("true_NZ", QPHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        self.zgrid = None
        self.bincents = None

    def _setup_iterator(self):

        itrs = [
            self.input_iterator('input'),
            self.input_iterator('tomography_bins'),
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
                        mask = np.ones(e-s, dtype=bool)
                    else:
                        mask = d['class_id'] == self.config.selected_bin
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
            self._process_chunk(
                s, e, data, mask, first, single_hist
            )
            first = False
        if self.comm is not None:  # pragma: no cover
            single_hist = self.comm.reduce(single_hist)

        if self.rank == 0:
            qp_d = qp.Ensemble(
                qp.hist, data=dict(bins=self.zgrid, pdfs=np.atleast_2d(single_hist))
            )
            self.add_data("true_NZ", qp_d)

    def _process_chunk(
        self, _start, _end, data, mask, _first, single_hist,
    ):
        squeeze_mask = np.squeeze(mask)
        zb = data[self.config.redshift_col][squeeze_mask]
        single_hist += np.histogram(zb, bins=self.zgrid)[0]
