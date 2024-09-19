import numpy as np
import scipy.special

from rail.utils.testing_utils import one_algo
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
    
    
    
def test_train_pz_with_wrong_columns():
    
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    datapath_pq = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )

    # ! create training data to be a data handle with path only
    # ! however it seems that with set_data() one always reads in the data
    training_data = DS.add_handle("pq", PqHandle, path=datapath_pq)
    
    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname=None,
        model="model_train_z.tmp",
        redshift_col="REDSHIFT",
    )
    
    train_algo = train_z.TrainZInformer
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz._get_stage_columns()
    with pytest.raises(KeyError):
        train_pz._check_column_names(training_data, train_pz.stage_columns)
    