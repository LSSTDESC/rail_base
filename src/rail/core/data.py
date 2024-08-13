"""Rail-specific data management"""

import os
import pickle
import enum
import tables_io
import qp

from .model import Model


class DataHandle:  # pylint: disable=too-many-instance-attributes
    """Class to act as a handle for a bit of data.  Associating it with a file and
    providing tools to read & write it to that file

    Parameters
    ----------
    tag : str
        The tag under which this data handle can be found in the store
    data : any or None
        The associated data
    path : str or None
        The path to the associated file
    creator : str or None
        The name of the stage that created this data handle
    """

    suffix = ""

    # This is to keep track of all the sub-types
    data_handle_type_dict = {}

    def __init__(self, tag, data=None, path=None, creator=None):
        """Constructor"""
        self.tag = tag
        if data is not None:
            self._validate_data(data)
        self.data = data
        self.path = path
        self.creator = creator
        self.fileObj = None
        self.groups = None
        self.partial = False
        self.length = None

    def __init_subclass__(cls, **kwargs):
        """Register the subclass with the dict"""
        cls.data_handle_type_dict[cls.__name__] = cls

    @classmethod
    def get_sub_class(cls, class_name):
        """Get a particular subclass by name"""
        return cls.data_handle_type_dict[class_name]

    @classmethod
    def print_sub_classes(cls):
        """Print the list of all the subclasses"""
        for key, val in cls.data_handle_type_dict.items():
            print(f"{key}: {val}")

    def open(self, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("DataHandle.open() called but path has not been specified")
        self.fileObj = self._open(os.path.expandvars(self.path), **kwargs)
        return self.fileObj

    @classmethod
    def _open(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._open")  # pragma: no cover

    def close(self, **kwargs):  # pylint: disable=unused-argument
        """Close"""
        self.fileObj = None

    def read(self, force=False, **kwargs):
        """Read and return the data from the associated file"""
        if self.data is not None and not force:
            return self.data
        self.set_data(self._read(os.path.expandvars(self.path), **kwargs))
        return self.data

    def __call__(self, **kwargs):
        """Return the data, re-reading the fill if needed"""
        if self.has_data and not self.partial:
            return self.data
        return self.read(force=True, **kwargs)

    @classmethod
    def _read(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._read")  # pragma: no cover

    def write(self, **kwargs):
        """Write the data to the associated file"""
        if self.path is None:
            raise ValueError(
                "TableHandle.write() called but path has not been specified"
            )
        if self.data is None:
            raise ValueError(
                f"TableHandle.write() called for path {self.path} with no data"
            )
        outdir = os.path.dirname(os.path.abspath(os.path.expandvars(self.path)))
        if not os.path.exists(outdir):  # pragma: no cover
            os.makedirs(outdir, exist_ok=True)
        return self._write(self.data, os.path.expandvars(self.path), **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        raise NotImplementedError("DataHandle._write")  # pragma: no cover

    def initialize_write(self, data_length, **kwargs):
        """Initialize file to be written by chunks"""
        if self.path is None:  # pragma: no cover
            raise ValueError(
                "TableHandle.write() called but path has not been specified"
            )
        self.groups, self.fileObj = self._initialize_write(
            self.data, os.path.expandvars(self.path), data_length, **kwargs
        )

    @classmethod
    def _initialize_write(cls, data, path, data_length, **kwargs):
        raise NotImplementedError("DataHandle._initialize_write")  # pragma: no cover

    def write_chunk(self, start, end, **kwargs):
        """Write the data to the associated file"""
        if self.data is None:
            raise ValueError(
                f"TableHandle.write_chunk() called for path {self.path} with no data"
            )
        if self.fileObj is None:
            raise ValueError(
                f"TableHandle.write_chunk() called before open for {self.tag} : {self.path}"
            )
        return self._write_chunk(
            self.data, self.fileObj, self.groups, start, end, **kwargs
        )

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        raise NotImplementedError("DataHandle._write_chunk")  # pragma: no cover

    def finalize_write(self, **kwargs):
        """Finalize and close file written by chunks"""
        if self.fileObj is None:  # pragma: no cover
            raise ValueError(
                f"TableHandle.finalize_wite() called before open for {self.tag} : {self.path}"
            )
        self._finalize_write(self.data, self.fileObj, **kwargs)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        raise NotImplementedError("DataHandle._finalize_write")  # pragma: no cover

    def iterator(self, **kwargs):
        """Iterator over the data"""
        if self.data is not None and self.partial is False:
            return self._in_memory_iterator(**kwargs)
        return self._iterator(self.path, **kwargs)

    def set_data(self, data, partial=False):
        """Set the data for a chunk, and set the partial flag to true"""
        self._validate_data(data)
        self.data = data
        self.partial = partial

    @classmethod
    def _validate_data(cls, data):  # pylint: disable=unused-argument
        """Make sure that the right type of data is being passed in"""
        return

    def size(self, **kwargs):
        """Return the size of the data associated to this handle"""
        if self.partial:
            if self.length is not None:  # pragma: no cover
                return self.length
            return self._size(self.path, **kwargs)
        if self.data is not None:
            return self.data_size(**kwargs)
        return self._size(self.path, **kwargs)

    def _size(self, path, **kwargs):
        raise NotImplementedError("DataHandle._size")  # pragma: no cover

    def data_size(self, **kwargs):
        """Return the size of the in memorry data"""
        if self.data is None:  # pragma: no cover
            return 0
        return self._data_size(self.data, **kwargs)

    def _data_size(self, data, **kwargs):
        raise NotImplementedError("DataHandle._data_size")  # pragma: no cover

    def _in_memory_iterator(self, **kwargs):
        raise NotImplementedError("DataHandle._in_memory_iterator")  # pragma: no cover

    @classmethod
    def _iterator(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._iterator")  # pragma: no cover

    @property
    def has_data(self):
        """Return true if the data for this handle are loaded"""
        return self.data is not None

    @property
    def has_path(self):
        """Return true if the path for the associated file is defined"""
        return self.path is not None

    @property
    def is_written(self):
        """Return true if the associated file has been written"""
        if self.path is None:
            return False
        return os.path.exists(os.path.expandvars(self.path))

    def __str__(self):
        s = f"{type(self)} "
        if self.has_path:
            s += f"{self.path}, ("
        else:
            s += "None, ("
        if self.is_written:
            s += "w"
        if self.has_data:
            s += "d"
        s += ")"
        return s

    @classmethod
    def make_name(cls, tag):
        """Construct and return file name for a particular data tag"""
        if cls.suffix:
            return f"{tag}.{cls.suffix}"
        return tag  # pragma: no cover


class TableHandle(DataHandle):
    """DataHandle for single tables of data"""

    suffix = None

    def set_data(self, data, partial=False):
        """Set the data for a chunk, and set the partial flag to true"""
        self._validate_data(data)
        self.data = data
        self.partial = partial

    @classmethod
    def _validate_data(cls, data):  # pylint: disable=unused-argument
        """Make sure that the right type of data is being passed in"""
        return

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file"""
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associated file"""
        return tables_io.write(data, path, **kwargs)

    def _size(self, path, **kwargs):
        if path in [None, "none", "None"]:  # pragma: no cover
            return 0
        return tables_io.io.getInputDataLength(path, **kwargs)

    def _data_size(self, data, **kwargs):
        group_name = kwargs.get("groupname", None)
        if group_name:
            try:
                data = data[group_name]
            except KeyError:  # pragma: no cover
                pass
        max_l = 0
        for _k, v in data.items():
            max_l = max(max_l, len(v))
        return max_l

    def _in_memory_iterator(self, **kwargs):
        nrows = self.data_size(**kwargs)
        groupname = kwargs.get("groupname", None)
        if isinstance(self.data, dict) and groupname:
            try:
                table = self.data[groupname]
            except KeyError:  # pragma: no cover
                table = self.data
        else:
            table = self.data
        for start, end in tables_io.ioUtils.data_ranges_by_rank(
            nrows, kwargs.get("chunk_size", 100000), 1, 0
        ):
            if isinstance(self.data, dict):
                yield start, end, tables_io.arrayUtils.sliceDict(
                    table, slice(start, end)
                )
            else:  # pragma: no cover
                yield start, end, table[start:end]

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        return tables_io.iteratorNative(path, **kwargs)


class Hdf5Handle(TableHandle):  # pragma: no cover
    """DataHandle for a table written to HDF5"""

    suffix = "hdf5"

    @classmethod
    def _initialize_write(cls, data, path, data_length, **kwargs):
        initial_dict = cls._get_allocation_kwds(data, data_length)
        comm = kwargs.get("communicator", None)
        group, fout = tables_io.io.initializeHdf5WriteSingle(
            path, groupname=None, comm=comm, **initial_dict
        )
        return group, fout

    @classmethod
    def _get_allocation_kwds(cls, data, data_length):
        keywords = {}
        for key, array in data.items():
            shape = list(array.shape)
            if not shape:
                continue
            shape[0] = data_length
            keywords[key] = (shape, array.dtype)
        return keywords

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        tables_io.io.writeDictToHdf5ChunkSingle(fileObj, data, start, end, **kwargs)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        return tables_io.io.finalizeHdf5Write(fileObj, **kwargs)


class FitsHandle(TableHandle):
    """DataHandle for a table written to fits"""

    suffix = "fits"


class PqHandle(TableHandle):
    """DataHandle for a parquet table"""

    suffix = "pq"

    def _size(self, path, **kwargs):
        return tables_io.io.getInputDataLengthPq(path, **kwargs)


class QPHandle(DataHandle):
    """DataHandle for qp ensembles"""

    suffix = "hdf5"

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file"""
        return qp.read(path)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associated file"""
        return data.write_to(path)

    @classmethod
    def _initialize_write(cls, data, path, data_length, **kwargs):
        comm = kwargs.get("communicator", None)
        return data.initializeHdf5Write(path, data_length, comm)

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        return data.writeHdf5Chunk(fileObj, start, end)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        return data.finalizeHdf5Write(fileObj)

    @classmethod
    def _validate_data(cls, data):
        if not isinstance(data, qp.Ensemble):
            raise TypeError(
                f"Expected `data` to be a `qp.Ensemble`, but {type(data)} was provided."
                "Perhaps you meant to use `TableHandle`?"
            )

    def _size(self, path, **kwargs):
        if self.data is not None and not self.partial:  # pragma: no cover
            return self._data_size(self.data)
        if path in [None, "none", "None"]:  # pragma: no cover
            return 0
        return tables_io.io.getInputDataLengthHdf5(path, groupname="data")

    def _data_size(self, data, **kwargs):
        return self.data.npdf

    def _in_memory_iterator(self, **kwargs):
        nrows = self.data.npdf
        for start, end in tables_io.ioUtils.data_ranges_by_rank(
            nrows, kwargs.get("chunk_size", 100000), 1, 0
        ):
            yield start, end, self.data[start:end]

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        kwargs.pop("groupname", "None")
        return qp.iterator(path, **kwargs)


class QPDictHandle(DataHandle):
    """DataHandle for dictionaries of qp ensembles"""

    suffix = "hdf5"

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the dictionary of qp.Ensembles from the associated file"""
        return qp.read_dict(path)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data (a dictionary of qp.Ensembles) to the associated file"""
        return qp.write_dict(path, data)


class QPOrTableHandle(QPHandle, Hdf5Handle):
    """DataHandle that should work with either qp.ensembles or tables"""

    suffix = "hdf5"

    class PdfOrValue(enum.Enum):
        unknown = -1
        distribution = 0
        point_estimate = 1
        both = 2

        def has_dist(self):
            return self.value in [0, 2]

        def has_point(self):
            return self.value in [1, 2]

    def is_qp(self):
        """Check if the associated data or file is a QP ensemble"""
        if self.path in [None, "None", "none"]:
            return isinstance(self.data, qp.Ensemble)
        return qp.is_qp_file(self.path)

    def check_pdf_or_point(self):
        """Check the associated file to see if it is a QP pdf, point estimate or both"""
        if self.is_qp():
            return self.PdfOrValue.both
        return self.PdfOrValue.point_estimate

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  # pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file"""
        if qp.is_qp_file(path):
            return qp.read(path, **kwargs)
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associated file"""
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _initialize_write(cls, data, path, data_length, **kwargs):
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        raise RuntimeError(
            "QPOrTableHandle should be used for input, not output"
        )  # pragma: no cover

    @classmethod
    def _validate_data(cls, data):
        pass

    def _size(self, path, **kwargs):
        if self.is_qp():
            return QPHandle._size(self, path, **kwargs)
        return Hdf5Handle._size(self, path, **kwargs)

    def _data_size(self, data, **kwargs):
        if self.is_qp():
            return self.data.npdf
        return Hdf5Handle._data_size(self, data, **kwargs)

    def _in_memory_iterator(self, **kwargs):
        if self.is_qp():
            return QPHandle._in_memory_iterator(self, **kwargs)
        return Hdf5Handle._in_memory_iterator(self, **kwargs)

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        if qp.is_qp_file(path):
            return QPHandle._iterator(path, **kwargs)
        return Hdf5Handle._iterator(path, **kwargs)


def default_model_read(modelfile):
    """Default function to read model files, simply used pickle.load"""
    with open(modelfile, 'rb') as fin:
        read_data = pickle.load(fin)
    if isinstance(read_data, Model):
        ret_data = read_data.data
    else:
        ret_data = read_data
    return ret_data


def default_model_write(model, path):
    """Write the model, this default implementation uses pickle"""
    with open(path, "wb") as fout:
        pickle.dump(obj=model, file=fout, protocol=pickle.HIGHEST_PROTOCOL)


class ModelDict(dict):
    """
    A specialized dict to keep track of individual estimation models objects:
    this is just a dict these additional features

    1. Keys are paths
    2. There is a read(path, force=False) method that reads a model object and
    inserts it into the dictionary
    3. There is a single static instance of this class
    """

    def open(self, path, mode, **kwargs):
        """Open the file and return the file handle"""
        return open(path, mode, **kwargs)  # pylint: disable=unspecified-encoding

    def read(
        self, path, force=False, reader=None, **kwargs
    ):  # pylint: disable=unused-argument
        """Read a model into this dict"""
        if reader is None:
            reader = default_model_read
        if force or path not in self:
            model = reader(path)
            self[path] = model
            return model
        return self[path]

    def write(
        self, model, path, force=False, writer=None, **kwargs
    ):  # pylint: disable=unused-argument
        """Write the model, this default implementation uses pickle"""
        if writer is None:
            writer = default_model_write
        if force or path not in self or not os.path.exists(path):
            self[path] = model
            writer(model, path)


class ModelHandle(DataHandle):
    """DataHandle for machine learning models"""

    suffix = "pkl"

    model_factory = ModelDict()

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file"""
        kwcopy = kwargs.copy()
        if kwcopy.pop("mode", "r") == "w":
            return cls.model_factory.open(path, mode="wb", **kwcopy)
        return cls.model_factory.read(path, **kwargs)

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file"""
        return cls.model_factory.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associated file"""
        return cls.model_factory.write(data, path, **kwargs)


class DataStore(dict):
    """Class to provide a transient data store

    This class:
    1) associates data products with keys
    2) provides functions to read and write the various data produces to associated files
    """

    allow_overwrite = False

    def __init__(self, **kwargs):
        """Build from keywords

        Note
        ----
        All of the values must be data handles of this will raise a TypeError
        """
        dict.__init__(self)
        for key, val in kwargs.items():
            self[key] = val

    def __str__(self):
        """Override __str__ casting to deal with `TableHandle` objects in the map"""
        s = "{"
        for key, val in self.items():
            s += f"  {key}:{val}\n"
        s += "}"
        return s

    def __repr__(self):
        """A custom representation"""
        s = "DataStore\n"
        s += self.__str__()
        return s

    def __setitem__(self, key, value):
        """Override the __setitem__ to work with ``TableHandle``"""
        if not isinstance(value, DataHandle):
            raise TypeError(
                f"Can only add objects of type DataHandle to DataStore, not {type(value)}"
            )
        check = self.get(key)
        if check is not None and not self.allow_overwrite:
            raise ValueError(
                f"DataStore already has an item with key {key},"
                "of type {type(check)}, created by {check.creator}"
            )
        dict.__setitem__(self, key, value)
        return value

    def __getattr__(self, key):
        """Allow attribute-like parameter access"""
        try:
            return self.__getitem__(key)
        except KeyError as msg:
            # Kludge to get docstrings to work
            if key in ["__objclass__"]:  # pragma: no cover
                return None
            raise KeyError from msg

    def __setattr__(self, key, value):
        """Allow attribute-like parameter setting"""
        return self.__setitem__(key, value)

    def add_data(self, key, data, handle_class, path=None, creator="DataStore"):
        """Create a handle for some data, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=data, creator=creator)
        self[key] = handle
        return handle

    def add_handle(self, key, handle_class, path, creator="DataStore"):
        """Create a handle for some data, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        self[key] = handle
        return handle

    def read_file(self, key, handle_class, path, creator="DataStore", **kwargs):
        """Create a handle, use it to read a file, and insert it into the DataStore"""
        handle = handle_class(key, path=path, data=None, creator=creator)
        handle.read(**kwargs)
        self[key] = handle
        return handle

    def read(self, key, force=False, **kwargs):
        """Read the data associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to read data {key} because {msg}") from msg
        return handle.read(force, **kwargs)

    def open(self, key, mode="r", **kwargs):
        """Open and return the file associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to open data {key} because {msg}") from msg
        return handle.open(mode=mode, **kwargs)

    def write(self, key, **kwargs):
        """Write the data associated to a particular key"""
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to write data {key} because {msg}") from msg
        return handle.write(**kwargs)

    def write_all(self, force=False, **kwargs):
        """Write all the data in this DataStore"""
        for key, handle in self.items():
            local_kwargs = kwargs.get(key, {})
            if handle.is_written and not force:
                continue
            handle.write(**local_kwargs)


_DATA_STORE = DataStore()


def DATA_STORE():
    """Return the factory instance"""
    return _DATA_STORE
