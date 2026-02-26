import os

import numpy as np
import pytest

from rail.core.data import QPHandle
from rail.core.stage import RailStage
from rail.estimation.algos.equal_count import EqualCountClassifier
from rail.estimation.algos.uniform_binning import UniformBinningClassifier
from rail.utils.path_utils import RAILDIR

# DS = RailStage.data_store
# DS.__class__.allow_overwrite = True

inputdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.hdf5")


@pytest.mark.parametrize(
    "input_param",
    [
        {"zbin_edges": [0.0, 0.3]},
        {"zmin": 0.0, "zmax": 0.3, "n_tom_bins": 1},
        {"zbin_edges": [0.0, 0.3], "object_id_col": "CATAID"},
    ],
)
def test_UniformBinningClassifier(input_param: dict) -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data_handle = QPHandle("input_data", path=inputdata)

    tomo = UniformBinningClassifier.make_stage(
        point_estimate_key="zmode",
        no_assign=-99,
        **input_param,
    )

    _out_data = tomo.classify(input_data_handle)
    os.remove(tomo.get_output(tomo.get_aliased_tag("output"), final_name=True))


def test_UniformBinningClassifier_binsize() -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data = QPHandle("input_data", path=inputdata)
    input_data.read()  # make sure the data is read in

    tomo = UniformBinningClassifier.make_stage(
        point_estimate_key="zmode",
        no_assign=-99,
        zmin=0.0,
        zmax=2.0,
        n_tom_bins=2,
    )
    output_data = tomo.classify(input_data)
    out_data = output_data.data

    # check length:
    assert len(out_data["class_id"]) == len(out_data["row_index"])

    # check that the assignment is as expected:
    assert (np.isin(np.unique(out_data["class_id"]), [1, 2, -99])).all()

    zb = input_data.data.ancil["zmode"]
    if 1 in out_data["class_id"]:
        assert (
            (zb[out_data["class_id"] == 1] >= 0.0)
            & (zb[out_data["class_id"] == 1] < 1.0)
        ).all()
    if 2 in out_data["class_id"]:
        assert (
            (zb[out_data["class_id"] == 2] >= 1.0)
            & (zb[out_data["class_id"] == 2] < 2.0)
        ).all()
    if -99 in out_data["class_id"]:
        assert (
            (zb[out_data["class_id"] == -99] < 0.0)
            | (zb[out_data["class_id"] == -99] >= 2.0)
        ).all()
    os.remove(tomo.get_output(tomo.get_aliased_tag("output"), final_name=True))


def test_UniformBinningClassifier_ancil() -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data_handle = QPHandle("input_data", path=inputdata)
    input_data = input_data_handle.read()

    tomo = UniformBinningClassifier.make_stage(
        point_estimate_key="zmedian",
        no_assign=-99,
        zmin=0.0,
        zmax=2.0,
        n_tom_bins=2,
    )
    with pytest.raises(KeyError):
        _out_data = tomo.classify(input_data)


@pytest.mark.parametrize(
    "input_param",
    [
        {"zmin": 0.0, "zmax": 0.3, "n_tom_bins": 1},
        {"zmin": 0.0, "zmax": 0.3, "n_tom_bins": 1, "object_id_col": "CATAID"},
    ],
)
def test_EqualCountClassifier(input_param: dict) -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data_handle = QPHandle("input_data", path=inputdata)
    input_data = input_data_handle.read()

    tomo = EqualCountClassifier.make_stage(
        point_estimate_key="zmode",
        no_assign=-99,
        **input_param,
    )

    _out_data = tomo.classify(input_data)
    os.remove(tomo.get_output(tomo.get_aliased_tag("output"), final_name=True))


def test_EqualCountClassifier_nobj() -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data_handle = QPHandle("input_data", path=inputdata)
    input_data = input_data_handle.read()

    tomo = EqualCountClassifier.make_stage(
        point_estimate_key="zmode",
        no_assign=-99,
        zmin=0.0,
        zmax=2.0,
        n_tom_bins=2,
    )
    output_data = tomo.classify(input_data)
    out_data = output_data.data

    # check that there are equal number of object in each bin modulo Ngal%Nbins
    assert (np.isin(np.unique(out_data["class_id"]), [1, 2, -99])).all()

    Ngal = sum(out_data["class_id"] != -99)
    exp_Ngal_perbin = int(Ngal / 2)
    # check that each bin does contain number of objects consistent with expected number
    # exp_Ngal_perbin + 1 to account for the cases where Ngal%Nbins!=0
    assert sum(out_data["class_id"] == 1) in [exp_Ngal_perbin, exp_Ngal_perbin + 1]
    assert sum(out_data["class_id"] == 2) in [exp_Ngal_perbin, exp_Ngal_perbin + 1]

    # check no assignment is correct
    if Ngal < len(out_data["class_id"]):
        zb = input_data.data.ancil["zmode"]
        assert (
            (zb[out_data["class_id"] == -99] < 0.0)
            | (zb[out_data["class_id"] == -99] >= 2.0)
        ).all()
    os.remove(tomo.get_output(tomo.get_aliased_tag("output"), final_name=True))


def test_EqualCountClassifier_ancil() -> None:
    # DS.clear()
    # input_data = DS.read_file("input_data", QPHandle, inputdata)
    input_data_handle = QPHandle("input_data", path=inputdata)
    input_data = input_data_handle.read()

    tomo = EqualCountClassifier.make_stage(
        point_estimate_key="zmedian",
        no_assign=-99,
        zmin=0.0,
        zmax=2.0,
        n_tom_bins=2,
    )
    with pytest.raises(KeyError):
        _out_data = tomo.classify(input_data)
