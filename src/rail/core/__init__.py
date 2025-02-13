"""Core code for RAIL"""

from .introspection import RailEnv
from .stage import RailPipeline, RailStage

__all__ = [
    "RailPipeline",
    "RailStage",
    "RailEnv",
]


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
