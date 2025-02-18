""" Parameters that are shared between stages """

from typing import Any

from ceci.config import StageConfig
from ceci.config import StageParameter as Param

lsst_bands = "ugrizy"
lsst_mag_cols = [f"mag_{band}_lsst" for band in lsst_bands]
lsst_mag_err_cols = [f"mag_err_{band}_lsst" for band in lsst_bands]
lsst_def_maglims = dict(
    mag_u_lsst=27.79,
    mag_g_lsst=29.04,
    mag_r_lsst=29.06,
    mag_i_lsst=28.62,
    mag_z_lsst=27.98,
    mag_y_lsst=27.05,
)
# default reddening parameters for LSST
lsst_def_a_env = dict(
    mag_u_lsst=4.81,
    mag_g_lsst=3.64,
    mag_r_lsst=2.70,
    mag_i_lsst=2.06,
    mag_z_lsst=1.58,
    mag_y_lsst=1.31,
)


SHARED_PARAMS = StageConfig(
    hdf5_groupname=Param(
        str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"
    ),
    chunk_size=Param(
        int, 10000, msg="Number of object per chunk for parallel processing"
    ),
    zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
    zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
    nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
    dz=Param(float, 0.01, msg="delta z in grid"),
    nondetect_val=Param(
        float, 99.0, msg="value to be replaced with magnitude limit for non detects"
    ),
    bands=Param(
        list, lsst_mag_cols, msg="Names of columns for magnitgude by filter band"
    ),
    err_bands=Param(
        list,
        lsst_mag_err_cols,
        msg="Names of columns for magnitgude errors by filter band",
    ),
    mag_limits=Param(dict, lsst_def_maglims, msg="Limiting magnitdues by filter"),
    band_a_env=Param(dict, lsst_def_a_env, msg="Redenning parameters"),
    ref_band=Param(str, "mag_i_lsst", msg="band to use in addition to colors"),
    redshift_col=Param(str, "redshift", msg="name of redshift column"),
    calculated_point_estimates=Param(
        dtype=list,
        default=[],
        msg="List of strings defining which point estimates to automatically calculate using `qp.Ensemble`."
        "Options include, 'mean', 'mode', 'median'.",
    ),
    recompute_point_estimates=Param(
        dtype=bool,
        default=False,
        msg="Force recomputation of point estimates",
    ),
)


class SharedParams:
    """Class to store parameters shared between many stages"""

    _config_text = SHARED_PARAMS.numpy_style_help_text()
    __doc__: str | None = f"\n\nParameters\n----------\n{_config_text}"

    @staticmethod
    def copy_param(param_name: str) -> Param:
        """Return a copy of one of the shared parameters

        Parameters
        ----------
        param_name
            Name of the parameter to copy

        Returns
        -------
        Param
            Copied parameter
        """
        return SHARED_PARAMS.get(param_name).copy()

    @staticmethod
    def set_param_default(param_name: str, default_value: Any) -> None:
        """Change the default value of one of the shared parameters

        Parameters
        ----------
        param_name
            Name of the parameter to copy

        default_value
            New default value
        """
        try:
            SHARED_PARAMS.get(param_name).set_default(default_value)
        except AttributeError as msg:  # pragma: no cover
            raise KeyError(
                f"No shared parameter {param_name} in SHARED_PARAMS"
            ) from msg

    @staticmethod
    def set_param_defaults(**kwargs: Any) -> None:  # pragma: no cover
        """Change the default value of several of the shared parameters

        Parameters
        ----------
        **kwargs
            Key, value pairs of parameter names and default values
        """
        for key, val in kwargs.items():
            set_param_default(key, val)


copy_param = SharedParams.copy_param

set_param_default = SharedParams.set_param_default

set_param_defaults = SharedParams.set_param_defaults
