"""Rail-specific data management"""

from __future__ import annotations

from typing import Any, TypeVar

from deprecated import deprecated

from .model import Model
from .data_types import DataLike, GroupLike, ModelLike, TableLike, FileLike
from .handle import (
    DataHandle,
    TableHandle,
    Hdf5Handle,
    FitsHandle,
    PqHandle,
    QPHandle,
    QPDictHandle,
    QPOrTableHandle,
    ModelHandle,
)

T = TypeVar("T", bound="DataHandle")


DEPRECATION_WARNING = (
    "DataStore interface is simplified and no longer saves a global dictionary"
)

__all__ = [
    "DataLike",
    "GroupLike",
    "ModelLike",
    "TableLike",
    "FileLike",
    "Model", 
    "DataHandle",
    "TableHandle",
    "Hdf5Handle",
    "FitsHandle",
    "PqHandle",
    "QPHandle",
    "QPDictHandle",
    "QPOrTableHandle",
    "ModelHandle",
]


class DataStore:
    """Class to provide a transient data store

    This class:

    1. associates data products with keys
    2. provides functions to read and write the various data produces to associated files

    """

    allow_overwrite = False

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def __str__(self) -> str:
        """Override __str__ casting to deal with `TableHandle` objects in the map"""
        return ""

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def __repr__(self) -> str:
        """A custom representation"""
        s = "DataStore\n"
        s += self.__str__()
        return s

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def __setitem__(self, key: str, value: DataHandle) -> None:
        """Override the __setitem__ to work with ``TableHandle``"""
        return

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def clear(self) -> None:
        """Placeholder for deprecated function"""
        return
    
    def add_data(
        self,
        key: str,
        data: DataLike,
        handle_class: type[DataHandle],
        path: str | None = None,
        creator: str = "DataStore",
    ) -> DataHandle:
        """Create a handle for some data, and insert it into the DataStore"""
        if path is None:
            path = f"{key}.{handle_class.suffix}"        
        handle = handle_class(key, path=path, data=data, creator=creator)
        handle.write()
        return handle

    def add_handle(
        self,
        key: str,
        handle_class: type[DataHandle],
        path: str,
        creator: str = "DataStore",
    ) -> DataHandle:
        """Create a handle for some data, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        return handle

    def read_file(
        self,
        key: str,
        handle_class: type[DataHandle],
        path: str,
        creator: str = "DataStore",
        **kwargs: Any,
    ) -> DataHandle:
        """Create a handle, use it to read a file, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        handle.read(**kwargs)
        return handle

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def read(self, key: str, force: bool = False, **kwargs: Any) -> DataLike:
        """Read the data associated to a particular key"""
        raise RuntimeError("DataStore.read(key) is deprecated")

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def open(self, key: str, mode: str = "r", **kwargs: Any) -> FileLike:
        """Open and return the file associated to a particular key"""
        raise RuntimeError("DataStore.read(key) is deprecated")

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def write(self, key: str, **kwargs: Any) -> None:
        """Write the data associated to a particular key"""
        raise RuntimeError("DataStore.write(key) is deprecated")

    @deprecated(version="2.0.0", reason=DEPRECATION_WARNING)
    def write_all(self, force: bool = False, **kwargs: Any) -> None:
        """Write all the data in this DataStore"""
        raise RuntimeError("DataStore.write_all() is deprecated")


_DATA_STORE = DataStore()


def DATA_STORE() -> DataStore:
    """Return the factory instance"""
    return _DATA_STORE
