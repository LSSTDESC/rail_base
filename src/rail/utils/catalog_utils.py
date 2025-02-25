from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from rail.core import common_params

T = TypeVar("T", bound="CatalogConfigBase")


class CatalogConfigBase:
    """Class that wraps the settings of shared configuration
    parameters needed work with the particular column names
    in a given catalog type
    """

    _sub_classes_by_class: dict[str, type[CatalogConfigBase]] = {}
    _sub_classes: dict[str, type[CatalogConfigBase]] = {}

    tag: str | None = None
    bandlist: str = ""
    maglims: list[float] = []
    a_env: list[float] = []
    hdf5_groupname: str = ""
    band_template: str = ""
    band_err_template: str = ""
    ref_band: str = ""
    redshift_col: str = ""

    _active_tag: str | None = None
    _active_class: type | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        assert cls.tag is not None
        cls._sub_classes_by_class[cls.__name__] = cls
        cls._sub_classes[cls.tag] = cls

    @classmethod
    def active_tag(cls) -> str | None:
        """Return the currently active tag"""
        return cls._active_tag

    @classmethod
    def active_class(cls: type[T]) -> type[T] | None:
        """Return the currently active class"""
        return cls._active_class

    @classmethod
    def subclasses_by_class(cls) -> dict[str, type[CatalogConfigBase]]:
        """Return the dict of all the sub-classes keyed by class name"""
        return cls._sub_classes_by_class

    @classmethod
    def subclasses(cls) -> dict[str, type[CatalogConfigBase]]:
        """Return the dict of all the sub-classes keyed by tag"""
        return cls._sub_classes

    @classmethod
    def get_class(cls, class_name: str, module_name: str) -> type[CatalogConfigBase]:
        """Return a class my name, loading it if needed"""
        if class_name not in cls._sub_classes_by_class:
            __import__(module_name)
        return cls._sub_classes_by_class[class_name]

    @classmethod
    def apply_class(cls, class_name: str) -> None:
        """Activate a particular class"""
        the_class = cls._sub_classes_by_class[class_name]
        cls._active_tag = the_class.tag
        cls._active_class = the_class
        cls._active_class._apply()

    @classmethod
    def apply(cls, tag: str) -> None:
        """Activate a particular tag"""
        cls._active_tag = tag
        cls._active_class = cls._sub_classes[tag]
        cls._active_class._apply()

    @classmethod
    def _apply(cls) -> None:
        base_dict = cls._build_base_dict()
        common_params.set_param_defaults(**base_dict)
        cls._apply_hook()

    @classmethod
    def _apply_hook(cls) -> None:
        return

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        """Retrun the mapping from band to column names"""
        return {band: cls.band_template.format(band=band) for band in cls.bandlist}

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        """Construct the mapping from band to limiting magnitude"""
        maglim_dict = {}
        for band, limx in zip(cls.bandlist, cls.maglims):
            maglim_dict[cls.band_template.format(band=band)] = limx
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        """Construct the mapping from band to a_env"""
        a_env_dict = {}
        for band, limx in zip(cls.bandlist, cls.a_env):
            a_env_dict[cls.band_template.format(band=band)] = limx
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        """Construct the list of column names for the filter magnitudes"""
        return [cls.band_template.format(band=band) for band in cls.bandlist]

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        """Construct the list of column names for uncertainties on the the filter magnitudes"""
        return [cls.band_err_template.format(band=band) for band in cls.bandlist]

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        """Contruct the name of the reference band"""
        return cls.band_template.format(band=ref_band)

    @classmethod
    def _build_base_dict(cls) -> dict:
        """Construct the dict of overrides for the shared paramters"""
        base_dict: dict = {}
        base_dict.update(
            bands=cls._build_band_names(),
            err_bands=cls._build_band_err_names(),
            mag_limits=cls._build_maglim_dict(),
            band_a_env=cls._build_a_env_dict(),
            ref_band=cls._build_ref_band(cls.ref_band),
            redshift_col=cls.redshift_col,
            nondetect_val=np.nan,
            hdf5_groupname=cls.hdf5_groupname,
        )
        return base_dict


class HscCatalogConfig(CatalogConfigBase):
    """Configuration for HSC data"""

    tag = "hsc"
    bandlist = "grizy"
    maglims = [27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "HSC{band}_cmodel_dered"
    band_err_template = "{band}_cmodel_magerr"
    ref_band = "i"
    redshift_col = "specz_redshift"
    hdf5_groupname = ""


class Dc2CatalogConfig(CatalogConfigBase):
    """Configuration for DC2 data"""

    tag = "dc2"
    bandlist = "ugrizy"
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "mag_{band}_lsst"
    band_err_template = "mag_err_{band}_lsst"
    ref_band = "i"
    redshift_col = "true_redshift"
    hdf5_groupname = ""


class RubinCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin project data"""

    tag = "rubin"
    bandlist = "ugrizy"
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    ref_band = "i"
    redshift_col = "true_redshift"
    hdf5_groupname = ""


class RomanRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Rubin data from Roman / Rubin simulations"""

    tag = "roman_rubin"
    bandlist = "ugrizy"
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    ref_band = "i"
    redshift_col = "redshift"
    hdf5_groupname = ""


class ComCamCatalogConfig(CatalogConfigBase):
    """Configuration for ComCam data"""

    tag = "com_cam"
    bandlist = "ugrizy"
    maglims = [26.4, 27.8, 27.1, 26.7, 25.8, 24.6]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "{band}_cModelMag"
    band_err_template = "{band}_cModelMagErr"
    ref_band = "i"
    redshift_col = "redshift"
    hdf5_groupname = ""


class RomanPlusRubinCatalogConfig(CatalogConfigBase):
    """Configuration for Roman + Rubin bands in Roman / Rubin simulations"""

    tag = "roman_plus_rubin"
    bandlist = "ugrizy"
    maglims = [24.0, 27.66, 27.25, 26.6, 26.24, 25.35]
    a_env = [4.81, 3.64, 2.70, 2.06, 1.58, 1.31]
    band_template = "LSST_obs_{band}"
    band_err_template = "LSST_obs_{band}_err"
    ref_band = "i"
    redshift_col = "redshift"
    hdf5_groupname = ""

    @classmethod
    def band_name_dict(cls) -> dict[str, str]:
        bands = super().band_name_dict()
        bands["f"] = "ROMAN_obs_F184"
        bands["h"] = "ROMAN_obs_H158"
        bands["j"] = "ROMAN_obs_J129"
        bands["y"] = "ROMAN_obs_Y106"
        return bands

    @classmethod
    def _build_maglim_dict(cls) -> dict[str, float]:
        maglim_dict = super()._build_maglim_dict()
        maglim_dict["ROMAN_obs_F184"] = 27.5
        maglim_dict["ROMAN_obs_H158"] = 28.1
        maglim_dict["ROMAN_obs_J129"] = 27.8
        maglim_dict["ROMAN_obs_Y106"] = 27.6
        return maglim_dict

    @classmethod
    def _build_a_env_dict(cls) -> dict[str, float]:
        a_env_dict = super()._build_a_env_dict()
        a_env_dict["ROMAN_obs_F184"] = 1.1
        a_env_dict["ROMAN_obs_H158"] = 1.2
        a_env_dict["ROMAN_obs_J129"] = 1.3
        a_env_dict["ROMAN_obs_Y106"] = 1.4
        return a_env_dict

    @classmethod
    def _build_band_names(cls) -> list[str]:
        bands = [cls.band_template.format(band=band) for band in cls.bandlist]
        bands += [
            "ROMAN_obs_F184",
            "ROMAN_obs_H158",
            "ROMAN_obs_J129",
            "ROMAN_obs_Y016",
        ]
        return bands

    @classmethod
    def _build_band_err_names(cls) -> list[str]:
        band_errs = [cls.band_err_template.format(band=band) for band in cls.bandlist]
        band_errs += [
            "ROMAN_obs_F184_err",
            "ROMAN_obs_H158_err",
            "ROMAN_obs_J129_err",
            "ROMAN_obs_Y106_err",
        ]
        return band_errs

    @classmethod
    def _build_ref_band(cls, ref_band: str = "i") -> str:
        return cls.band_template.format(band=ref_band)


apply_defaults = CatalogConfigBase.apply
