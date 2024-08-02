"""Rail-specific data management"""

from typing import Any
import pickle


class Model:
    """Class to act as wrapper for ML models """

    def __init__(
        self,
        data: Any,
        creation_class_name: str,
        version: int=0,
        provenance: dict[str, Any] | None=None,
    ):
        """Constructor

        Parameters
        ----------
        data : Any
            Model data
        creation_class_name : str
            Name of the creation class
        version : int
            Version of the model
        provenance: dict[str, Any] | None
            Provenance infomration
        """
        self.data = data
        self.creation_class_name = creation_class_name
        self.version = version
        if provenance is not None:
            self.provenance = provenance.copy()
        else:
            self.provenance = {}

    def validate(
        self,
        creation_class_name: str | None,
        version: int | None
    ) -> None:
        """
        Parameters
        ----------
        creation_class_name : str | None
            Name of the creation class
        version : int | None
            Version of the model

        Raises
        ------
        TypeError : Either creation_class_name or version does not match
        """
        if creation_class_name is not None and creation_class_name != self.creation_class_name:
            raise TypeError(f"Model.creation_class_name does not match.  {creation_class_name} != {self.creation_class_name}")
        if version is not None and version != self.version:
            raise TypeError(f"Model.version does not match.  {version} != {self.version}")

    @classmethod
    def read(
        cls,
        path: str,
        creation_class_name: str="dummy",
        version: int=0,
        provenance: dict[str, Any] | None = None,
    ):
        """Read a model from a file.

        Note that this will promote the data to a Model if it is not already

        Parameters
        ----------
        path: str,
            File to read
        creation_class_name : str
            Name of the creation class
        version : int
            Version of the model
        provenance: dict[str, Any] | None
            Provenance infomration

        Returns
        -------
        model: Model
            Newly read Model
        """
        with open(path, 'rb') as fin:
            read_data = pickle.load(fin)

        if isinstance(read_data, Model):
            return read_data

        if provenance is None:
            provenance = {}

        return cls(read_data, creation_class_name, version, provenance)

    @classmethod
    def wrap(
        cls,
        inpath: str,
        outpath: str,
        creation_class_name: str="dummy",
        version: int=0,
        provenance: dict[str, Any] | None = None,
    ):
        """Read a model from a file and write it as a `Model` if it is not already

        Parameters
        ----------
        inpath: str,
            File to read
        outpath: str,
            File to write
        creation_class_name : str
            Name of the creation class
        version : int
            Version of the model
        provenance: dict[str, Any] | None
            Provenance information

        Returns
        -------
        model: Model
            Newly read & converted Model
        """
        the_model = cls.read(inpath, creation_class_name, version, provenance)
        the_model.write(outpath)
        return the_model

    @classmethod
    def dump(
        cls,
        obj: Any,
        path: str,
        creation_class_name: str="dummy",
        version: int=0,
        provenance: dict[str, Any] | None = None,
    ):
        """Write an object to a model file

        This will promote it to a Model if it isn't already

        Parameters
        ----------
        obj: Any
            Object to dump
        path: str,
            File to write
        creation_class_name : str
            Name of the creation class
        version : int
            Version of the model
        provenance: dict[str, Any] | None
            Provenance information

        Returns
        -------
        model: Model
            Newly converted Model
        """

        if isinstance(obj, Model):
            write_obj = obj
        else:
            if provenance is None:
                provenance = {}
            write_obj = cls(obj, creation_class_name, version, provenance)

        write_obj.write(path)
        return write_obj

    def write(
        self,
        path: str,
    ) -> None:
        """Write a model to a file

        Parameters
        ----------
        path: str,
            File to write
        """
        with open(path, "wb") as fout:
            pickle.dump(obj=self, file=fout, protocol=pickle.HIGHEST_PROTOCOL)
