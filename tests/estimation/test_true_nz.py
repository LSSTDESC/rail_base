import os
import numpy as np
import pytest
import qp

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
    
    nz_hist = TrueNZHistogrammer.make_stage(
        name='true_nz',
        hdf5_groupname='photometry',
        redshift_col='redshift',
        zmin=0.0,
        zmax=3.0,
        nzbins=301,            
    )
    out_hist = nz_hist.histogram(true_nz, tomo_bins)

    check_ens = qp.read(out_hist.path)
    assert check_ens.ancil['n_total'][0] == 10

    check_vals = [0, 5, 0, 0, 0]
    
    for i in range(5):
        nz_hist = TrueNZHistogrammer.make_stage(
            name='true_nz',
            hdf5_groupname='photometry',
            redshift_col='redshift',
            zmin=0.0,
            zmax=3.0,
            nzbins=301,            
            selected_bin=i,
        )
        out_hist = nz_hist.histogram(true_nz, tomo_bins)
        check_ens = qp.read(out_hist.path)
        assert check_ens.ancil['n_total'][0] == check_vals[i]

        
