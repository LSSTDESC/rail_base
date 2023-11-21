"""Core code for RAIL"""
import pkgutil
import os
import setuptools
import rail

from .stage import RailPipeline, RailStage

# from .utilPhotometry import PhotormetryManipulator, HyperbolicSmoothing, HyperbolicMagnitudes
from .util_stages import ColumnMapper, RowSelector, TableConverter
from .introspection import RailEnv
from .point_estimation import PointEstimationMixin


def find_version():
    """Find the version"""
    # setuptools_scm should install a
    # file _version alongside this one.
    from . import _version  # pylint: disable=import-outside-toplevel

    return _version.version


try:
    __version__ = find_version()
except ImportError:  # pragma: no cover
    __version__ = "unknown"
