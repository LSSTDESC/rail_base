"""
Implementation of the 'pathological photo-z PDF estimator,
as used in arXiv:2001.03621 (see section 3.3). It assigns each test set galaxy
a photo-z PDF equal to the normalized redshift distribution
N (z) of the training set.
"""

import numpy as np
import qp

from rail.core.common_params import SHARED_PARAMS
from rail.estimation.estimator import CatEstimator
from rail.estimation.informer import CatInformer


class trainZmodel:
    """
    Temporary class to store the single trainZ pdf for trained model.
    Given how simple this is to compute, this seems like overkill.
    """

    def __init__(self, zgrid, pdf, zmode):
        self.zgrid = zgrid
        self.pdf = pdf
        self.zmode = zmode


class TrainZInformer(CatInformer):
    """Train an Estimator which returns a global PDF for all galaxies"""

    name = "TrainZInformer"
    config_options = CatInformer.config_options.copy()
    config_options.update(
        zmin=SHARED_PARAMS,
        zmax=SHARED_PARAMS,
        nzbins=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
    )

    def run(self):
        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")
        zbins = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins + 1)
        speczs = np.sort(training_data[self.config.redshift_col])
        train_pdf, _ = np.histogram(speczs, zbins)
        midpoints = zbins[:-1] + np.diff(zbins) / 2
        zmode = midpoints[np.argmax(train_pdf)]
        cdf = np.cumsum(train_pdf)
        cdf = cdf / cdf[-1]
        norm = cdf[-1] * (zbins[2] - zbins[1])
        train_pdf = train_pdf / norm
        zgrid = midpoints
        self.model = trainZmodel(zgrid, train_pdf, zmode)
        self.add_data("model", self.model)

    def validate(self):
        """Validation which checks if the required column names by the stage exist in the data"""
        self._get_stage_columns()
        data = self.get_handle("input", allow_missing=True)
        # **kwargs in the function below is omitted
        # as these params are not yet implemented
        self._check_column_names(data, self.stage_columns)

    def _get_stage_columns(self):
        self.stage_columns = [self.config.redshift_col]


class TrainZEstimator(CatEstimator):
    """CatEstimator which returns a global PDF for all galaxies"""

    name = "TrainZEstimator"
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS, zmax=SHARED_PARAMS, nzbins=SHARED_PARAMS)

    def __init__(self, args, **kwargs):
        self.zgrid = None
        self.train_pdf = None
        self.zmode = None
        CatEstimator.__init__(self, args, **kwargs)

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:  # pragma: no cover
            return
        self.zgrid = self.model.zgrid
        self.train_pdf = self.model.pdf
        self.zmode = self.model.zmode

    def _process_chunk(self, start, end, data, first):
        test_size = end - start
        zmode = np.repeat(self.zmode, test_size)
        qp_d = qp.Ensemble(
            qp.interp,
            data=dict(xvals=self.zgrid, yvals=np.tile(self.train_pdf, (test_size, 1))),
        )
        qp_d.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(qp_d, start, end, first)
