import os

import numpy as np
import pytest
import scipy.special

from rail.core.data import PqHandle, QPHandle, TableHandle
from rail.core.stage import RailStage
from rail.estimation.algos import gaussian_pz, random_gauss, train_z
from rail.utils.path_utils import RAILDIR
from rail.utils.testing_utils import one_algo

sci_ver_str = scipy.__version__.split(".")


# DS = RailStage.data_store
# DS.__class__.allow_overwrite = True


def test_random_pz() -> None:
    train_config_dict: dict = {}
    estim_config_dict = {
        "rand_width": 0.025,
        "rand_zmin": 0.0,
        "rand_zmax": 3.0,
        "calc_summary_stats": True,
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
    try:
        os.remove("model.pkl")
    except FileNotFoundError:  # pragma: no cover
        pass


def test_train_pz() -> None:
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

    assert (results.ancil["distribution_type"] == 0).all()
    assert np.isclose(results.ancil["zmode"], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()

    try:
        os.remove("model.pkl")
    except FileNotFoundError:  # pragma: no cover
        pass


def test_train_pz_with_wrong_columns_path() -> None:
    # DS.clear()
    # DS.__class__.allow_overwrite = False

    datapath_pq = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )

    # ! create training data to be a data handle with path only
    # ! however it seems that with set_data() one always reads in the data
    # ! hence the way we make the data here:
    training_data1 = PqHandle("input", path=datapath_pq)

    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname=None,
        model="model_train_z.tmp",
        input=training_data1,
        redshift_col="REDSHIFT",
    )

    train_algo = train_z.TrainZInformer
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz._get_stage_columns()
    with pytest.raises(KeyError):
        # testing the case where only path exist
        train_pz._check_column_names(training_data1, train_pz.stage_columns)


def test_train_pz_with_wrong_columns_nogroupname() -> None:
    # DS.clear()
    # DS.__class__.allow_overwrite = False

    datapath_pq = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )

    training_data2 = PqHandle("input", path=datapath_pq)

    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname=None,
        input=training_data2,
        model="model_train_z.tmp",
        redshift_col="REDSHIFT",
    )

    train_algo = train_z.TrainZInformer
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz._get_stage_columns()
    with pytest.raises(KeyError):
        # testing the case where data handle has no hdf5 group name
        train_pz._check_column_names(training_data2, train_pz.stage_columns)


def test_train_pz_with_wrong_columns_table() -> None:
    # DS.clear()
    # DS.__class__.allow_overwrite = False

    datapath_pq = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"
    )

    # ! create training data to be a table
    # ! however it seems that with set_data() one always reads in the data handle
    # ! hence the way we make the data here:
    training_data2 = PqHandle("input", path=datapath_pq)
    # training_data3 = training_data2.data
    training_data3 = training_data2.read()

    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname=None,
        input=datapath_pq,
        model="model_train_z.tmp",
        redshift_col="REDSHIFT",
    )

    train_algo = train_z.TrainZInformer
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz._get_stage_columns()
    with pytest.raises(KeyError):
        # testing the case where data is a table
        train_pz._check_column_names(training_data3, train_pz.stage_columns)


def test_train_pz_with_wrong_columns_table_wgroupname() -> None:
    # DS.clear()
    # DS.__class__.allow_overwrite = False

    traindata = os.path.join(
        RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5"
    )

    # ! create training data to be a table
    # ! however it seems that with set_data() one always reads in the data handle
    # ! hence the way we make the data here:
    training_data2 = TableHandle("input", path=traindata)
    # training_data3 = training_data2.data
    training_data3 = training_data2.read()

    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname="photometry",
        input=training_data2,
        model="model_train_z.tmp",
        redshift_col="REDSHIFT",
    )

    train_algo = train_z.TrainZInformer
    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz._get_stage_columns()
    with pytest.raises(KeyError):
        # testing the case where data is a table
        train_pz._check_column_names(training_data3, train_pz.stage_columns)


def test_gaussian_pz() -> None:

    # DS.clear()
    # DS.__class__.allow_overwrite = False

    input_path = os.path.join(
        RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5"
    )

    input_data = QPHandle("input_data", path=input_path)

    inform_gauss_pz = gaussian_pz.GaussianPzInformer.make_stage()
    model = inform_gauss_pz.inform(input_data)

    estimate_gauss_pz = gaussian_pz.GaussianPzEstimator.make_stage(model=model)
    outdata = estimate_gauss_pz.estimate(input_data)
    assert outdata.data.npdf == 10


def test_train_pz_with_output_mode_return():
    """Tests that output_mode = return works for estimation algorithms"""
    train_config_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        hdf5_groupname="photometry",
        model="model_train_z.tmp",
    )
    estim_config_dict = dict(
        hdf5_groupname="photometry", model="model_train_z.tmp", output_mode="return"
    )

    zb_expected = np.repeat(0.1445183, 10)
    pdf_expected = np.zeros(shape=(301,))
    pdf_expected[10:16] = [7, 23, 8, 23, 26, 13]
    train_algo = train_z.TrainZInformer
    pz_algo = train_z.TrainZEstimator
    results, rerun_results, _ = one_algo(
        "TrainZ", train_algo, pz_algo, train_config_dict, estim_config_dict
    )

    assert (results.ancil["distribution_type"] == 0).all()
    assert np.isclose(results.ancil["zmode"], zb_expected).all()
    assert np.isclose(results.ancil["zmode"], rerun_results.ancil["zmode"]).all()

    try:
        os.remove("model.pkl")
    except FileNotFoundError:  # pragma: no cover
        pass
