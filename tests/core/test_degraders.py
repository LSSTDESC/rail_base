import os

import numpy as np
import pandas as pd
import pytest

from rail.core.data import DATA_STORE, TableHandle
from rail.core.util_stages import ColumnMapper
from rail.creation.degradation.quantityCut import QuantityCut
from rail.creation.degradation.spectroscopic_selections import *  # pylint: disable=wildcard-import,unused-wildcard-import


@pytest.fixture(name='data')
def data_fixture():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data", df, TableHandle, path="dummy.pd")


@pytest.fixture(name='data_forspec')
def data_forspec_fixture():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(200000, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data_forspec", df, TableHandle, path="dummy_forspec.pd")


@pytest.mark.parametrize(
    "cuts,error",
    [
        (1, TypeError),
        ({"u": "cut"}, TypeError),
        ({"u": dict()}, TypeError),
        ({"u": [1, 2, 3]}, ValueError),
        ({"u": [1, "max"]}, TypeError),
        ({"u": [2, 1]}, ValueError),
        ({"u": TypeError}, TypeError),
    ],
)
def test_QuantityCut_bad_params(cuts, error):
    """Test bad parameters that should return Type and Value errors"""
    with pytest.raises(error):
        QuantityCut.make_stage(cuts=cuts)


def test_QuantityCut_returns_correct_shape(data):
    """Make sure QuantityCut is returning the correct shape"""

    cuts = {
        "u": 0,
        "y": (1, 2),
    }
    degrader = QuantityCut.make_stage(cuts=cuts)

    degraded_data = degrader(data).data

    assert degraded_data.shape == data.data.query("u < 0 & y > 1 & y < 2").shape
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


def test_SpecSelection(data):
    bands = ["u", "g", "r", "i", "z", "y"]
    _band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    _standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data = col_remapper_test(data)

    degrader_GAMA = SpecSelection_GAMA.make_stage()
    degrader_GAMA(data)
    repr(degrader_GAMA)

    os.remove(
        degrader_GAMA.get_output(
            degrader_GAMA.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_BOSS = SpecSelection_BOSS.make_stage()
    degrader_BOSS(data)
    repr(degrader_BOSS)

    os.remove(
        degrader_BOSS.get_output(
            degrader_BOSS.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_DEEP2 = SpecSelection_DEEP2.make_stage()
    degrader_DEEP2(data)
    repr(degrader_DEEP2)

    os.remove(
        degrader_DEEP2.get_output(
            degrader_DEEP2.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_VVDSf02 = SpecSelection_VVDSf02.make_stage()
    degrader_VVDSf02(data)
    repr(degrader_VVDSf02)

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(
        colnames={"i": "mag_i_lsst", "redshift": "redshift"}
    )
    degrader_zCOSMOS(data)
    repr(degrader_zCOSMOS)

    os.remove(
        degrader_zCOSMOS.get_output(
            degrader_zCOSMOS.get_aliased_tag("output"), final_name=True
        )
    )

    degrader_HSC = SpecSelection_HSC.make_stage()
    degrader_HSC(data)
    repr(degrader_HSC)

    os.remove(
        degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True)
    )

    degrader_HSC = SpecSelection_HSC.make_stage(percentile_cut=70)
    degrader_HSC(data)
    repr(degrader_HSC)

    os.remove(
        degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True)
    )


def test_SpecSelection_low_N_tot(data_forspec):
    bands = ["u", "g", "r", "i", "z", "y"]
    _band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    _standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data_forspec = col_remapper_test(data_forspec)

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(N_tot=1)
    degrader_zCOSMOS(data_forspec)

    os.remove(
        degrader_zCOSMOS.get_output(
            degrader_zCOSMOS.get_aliased_tag("output"), final_name=True
        )
    )


@pytest.mark.parametrize("N_tot, errortype", [(-1, ValueError)])
def test_SpecSelection_bad_params(N_tot, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(N_tot=N_tot)


@pytest.mark.parametrize("errortype", [(ValueError)])
def test_SpecSelection_bad_colname(data, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        degrader_GAMA = SpecSelection_GAMA.make_stage()
        degrader_GAMA(data)


@pytest.mark.parametrize(
    "success_rate_dir, errortype", [("/this/path/should/not/exist", ValueError)]
)
def test_SpecSelection_bad_path(success_rate_dir, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(success_rate_dir=success_rate_dir)
