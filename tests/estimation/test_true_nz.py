import os
import numpy as np
import pytest

from rail.utils.path_utils import RAILDIR
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.estimation.algos.true_nz import TrueNZHistogrammer


DS = RailStage.data_store
DS.__class__.allow_overwrite = True

true_nz_file = "src/rail/examples_data/testdata/validation_10gal.hdf5"
tomo_file = "src/rail/examples_data/testdata/output_tomo.hdf5"


def test_true_nz():
    DS.clear()
    true_nz = DS.read_file('true_nz', path=true_nz_file, handle_class=TableHandle)
    tomo_bins = DS.read_file('tomo_bins', path=tomo_file, handle_class=TableHandle)
    
    nz_hist = TrueNZHistogrammer.make_stage(name='true_nz', hdf5_groupname='photometry', redshift_col='redshift')
    out_hist = nz_hist.histogram(true_nz, tomo_bins)

    
    
