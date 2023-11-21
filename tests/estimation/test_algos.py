import numpy as np
import scipy.special

from rail.core.algo_utils import one_algo
from rail.core.stage import RailStage
from rail.estimation.algos import random_gauss, train_z

sci_ver_str = scipy.__version__.split(".")


DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def test_random_pz():
    train_config_dict = {}
    estim_config_dict = {
        "rand_width": 0.025,
        "rand_zmin": 0.0,
        "rand_zmax": 3.0,
        "nzbins": 301,
        "hdf5_groupname": "photometry",
        "model": "None",
        "seed": 42,
    }
    zb_expected = np.array(
        [2.322, 1.317, 2.576, 2.092, 0.283, 2.927, 2.283, 2.358, 0.384, 1.351]
    )
    train_algo = random_gauss.RandomGaussInformer
    pz_algo = random_gauss.RandomGaussEstimator
    results, _, _ = one_algo(
        "RandomPZ", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    assert np.isclose(results.ancil["zmode"], zb_expected).all()


def test_train_pz():
    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname="photometry",
        model="model_train_z.tmp",
    )
    estim_config_dict = dict(hdf5_groupname="photometry", model="model_train_z.tmp")

    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301,))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    train_algo = train_z.TrainZInformer
    pz_algo = train_z.TrainZEstimator
    results, rerun_results, _ = one_algo(
        "TrainZ", train_algo, pz_algo, train_config_dict, estim_config_dict
    )
    assert np.isclose(results.ancil["zmode"], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()
