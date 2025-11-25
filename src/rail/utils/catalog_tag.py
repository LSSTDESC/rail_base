from __future__ import annotations

from typing import Any

import numpy as np

from ceci.config import StageParameter as Param
from rail.core import common_params
from rail.core.configurable import Configurable


class Band(Configurable):

    config_options: dict[str, Param] = dict(
        name=Param(str, None, required=True, msg="Name for this band"),
        a_env=Param(float, None, required=True, msg="Reddening parameter"),
    )

    yaml_tag: str = "Band"

    _band_dict: dict[str, Band] = {}

    def __init__(self, **kwargs: Any) -> None:
        """C'tor

        Parameters
        ----------
        kwargs: Any
            Configuration parameters for this Band, must match
            class.config_options data members
        """
        Configurable.__init__(self, **kwargs)
        if self.config.name in self._band_dict:
            raise KeyError(f"Duplicate band {self.config.name}")
        self._band_dict[self.config.name] = self

    @classmethod
    def get_band(cls, key: str) -> Band:
        """Return a particular band"""
        return cls._band_dict[key]

    @classmethod
    def clear(cls) -> None:
        cls._band_dict.clear()


class CatalogTag(Configurable):

    config_options = dict(
        name=Param(
            str, None, required=True, msg="Tag to associate to this Catalog type"
        ),
        mag_column_template=Param(
            str,
            None,
            required=True,
            msg="Template to construct band magnitude column names",
        ),
        mag_err_column_template=Param(
            str,
            None,
            required=True,
            msg="Template to construct band magnitude error column names",
        ),
        filter_template=Param(
            str, None, required=True, msg="Template to filter file names"
        ),
        redshift_col=Param(str, "redshift", msg="Name of the redshift column"),
        object_id_col=Param(str, "objectId", msg="Name of the object id column"),
        nondetect_val=Param(float, np.nan, msg="Guard value for non-detections"),
        nonobserved_val=Param(float, np.inf, msg="Guard value for non-observations"),
        hdf5_groupname=Param(
            str, "", msg="HDF5 group with photometry information, blank for top-level"
        ),
        ref_band=Param(str, "i", msg="Reference band"),
        zp_error_default=Param(float, 0.1, msg="Default zero-point error"),
        replace_error_val_default=Param(float, 0.1, msg="Replacement error value"),
        bands=Param(dict, {}, msg="Band definitions"),
        band_list=Param(list, [], msg="Names of bands in order"),
    )

    yaml_tag: str = "CatalogTag"

    _tag_dict: dict[str, CatalogTag] = {}

    def __init__(self, **kwargs: Any) -> None:
        """C'tor

        Parameters
        ----------
        kwargs: Any
            Configuration parameters for this Band, must match
            class.config_options data members
        """
        Configurable.__init__(self, **kwargs)
        if self.config.name in self._tag_dict:
            raise KeyError(f"Duplicate CatalogTag {self.config.name}")
        self._tag_dict[self.config.name] = self
        self._base_dict: dict = {}

    def _build_base_dict(self) -> dict:
        """Construct the dict of overrides for the shared paramters"""
        set_params: list[str] = [
            "redshift_col",
            "nondetect_val",
            "nonobserved_val",
            "hdf5_groupname",
        ]
        base_dict: dict = {key: self.config[key] for key in set_params}
        shared_pars = self._build_shared_params()
        base_dict.update(
            ref_band=self._build_ref_band(),
            **shared_pars,
        )
        return base_dict

    @classmethod
    def clear(cls) -> None:
        cls._tag_dict.clear()

    def _build_shared_params(self) -> dict:
        """Build the various shared params"""

        bands: list[str] = []
        err_bands: list[str] = []
        err_dict: dict[str, str] = {}
        mag_limits: dict[str, float] = {}
        band_a_env: dict[str, float] = {}
        filter_list: list[str] = []
        zp_errors: list[float] = []
        replace_error_vals: list[float] = []

        for band_name in self.config.band_list:
            band_info: dict = self.config.bands[band_name]
            if "filter" in band_info:
                filter_name = band_info["filter"]
            else:
                filter_name = self.config.filter_template.format(band=band_name)
            filter_list.append(filter_name)
            band_val = Band.get_band(filter_name)
            band_a_env[band_name] = band_val.config.a_env
            if "mag_column_name" in band_info:
                mag_column_name = band_info["mag_column_name"]
            else:
                mag_column_name = self.config.mag_column_template.format(band=band_name)
            if "mag_err_column_name" in band_info:
                mag_err_column_name = band_info["mag_err_column_name"]
            else:
                mag_err_column_name = self.config.mag_err_column_template.format(
                    band=band_name
                )
            bands.append(mag_column_name)
            err_bands.append(mag_err_column_name)
            err_dict[mag_column_name] = mag_err_column_name
            mag_limits[band_name] = band_info["mag_limit"]
            if "zp_error" in band_info:
                zp_err = band_info["zp_error"]
            else:
                zp_err = self.config.zp_error_default
            zp_errors.append(zp_err)
            if "replace_error_val" in band_info:
                rev = band_info["replace_error_val"]
            else:
                rev = self.config.replace_error_val_default
            replace_error_vals.append(rev)

        return dict(
            bands=bands,
            err_bands=err_bands,
            err_dict=err_dict,
            mag_limits=mag_limits,
            band_a_env=band_a_env,
            filter_list=filter_list,
            zp_errors=zp_errors,
            replace_error_vals=replace_error_vals,
        )

    def _build_ref_band(self) -> str:
        """Contruct the name of the reference band"""
        return self.config.mag_column_template.format(band=self.config.ref_band)

    def apply_tag(self) -> None:
        """Apply this tag"""
        self._base_dict = self._build_base_dict()
        common_params.set_param_defaults(**self._base_dict)

    @classmethod
    def apply(cls, tag: str) -> None:
        """Activate a particular tag"""
        if tag not in cls._tag_dict:
            raise KeyError(
                f"Did not find tag: {tag} in known CatalogTags: {list(cls._tag_dict.keys())}"
            )
        cls._active_tag = tag
        tag_object = cls._tag_dict[tag]
        tag_object.apply_tag()
