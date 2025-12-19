"""Stages that implement utility functions"""

import tables_io
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle, PqHandle, TableLike
from rail.core.stage import RailStage


class ColumnMapper(RailStage):
    """Utility stage that remaps the names of columns.

    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data.rename(columns=self.config.columns, in_place=self.config.in_place)`

    """

    name = "ColumnMapper"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "column_mapper"
    config_options = RailStage.config_options.copy()
    config_options.update(
        columns=Param(dict, required=True, msg="Map of columns to rename"),
        in_place=Param(bool, default=False, msg="Update file in place"),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def run(self) -> None:
        data = self.get_data("input", allow_missing=True)
        out_data = data.rename(
            columns=self.config.columns, inplace=self.config.in_place
        )
        if self.config.in_place:  # pragma: no cover
            out_data = data
        self.add_data("output", out_data)

    def __repr__(self) -> str:  # pragma: no cover
        printMsg = "Stage that applies remaps the following column names in a pandas DataFrame:\n"
        printMsg += "f{str(self.config.columns)}"
        return printMsg

    def __call__(self, data: TableLike, **kwargs) -> PqHandle:
        """Return a table with the columns names changed

        Parameters
        ----------
        data : TableLike
            The data to be renamed

        Returns
        -------
        PqHandle
            The degraded sample
        """
        self.set_data("input", data)
        self.run()
        self.finalize()
        return self.get_handle("output")


class RowSelector(RailStage):
    """Utility Stage that sub-selects rows from a table by index

    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data[self.config.start_row:self.config.stop_row]`

    """

    name = "RowSelector"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "row_selector"
    config_options = RailStage.config_options.copy()
    config_options.update(
        start_row=Param(int, required=True, msg="starting row number"),
        stop_row=Param(int, required=True, msg="Stoppig row number"),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def run(self) -> None:
        data = self.get_data("input", allow_missing=True)
        out_data = data.iloc[self.config.start_row : self.config.stop_row]
        self.add_data("output", out_data)

    def __repr__(self) -> str:  # pragma: no cover
        printMsg = "Stage that applies remaps the following column names in a pandas DataFrame:\n"
        printMsg += "f{str(self.config.columns)}"
        return printMsg

    def __call__(self, data: TableLike, **kwargs) -> PqHandle:
        """Return a table with the columns names changed

        Parameters
        ----------
        data : TableLike
            The data to be renamed

        Returns
        -------
        PqHandle
            The degraded sample
        """
        self.set_data("input", data)
        self.run()
        self.finalize()
        return self.get_handle("output")


class TableConverter(RailStage):
    """Utility stage that converts tables from one format to anothe

    FIXME, this is hardwired to convert parquet tables to Hdf5Tables.
    It would be nice to have more options here.
    """

    name = "TableConverter"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "table_converter"
    config_options = RailStage.config_options.copy()
    config_options.update(
        output_format=Param(str, required=True, msg="Format of output table"),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", Hdf5Handle)]

    def run(self) -> None:
        data = self.get_data("input", allow_missing=True)
        out_fmt = tables_io.types.TABULAR_FORMAT_NAMES[self.config.output_format]
        out_data = tables_io.convert(data, out_fmt)
        self.add_data("output", out_data)

    def __call__(self, data: TableLike, **kwargs) -> Hdf5Handle:
        """Return a converted table

        Parameters
        ----------
        data : TableLike
            The data to be converted

        Returns
        -------
        Hdf5Handle
            The converted version of the table
        """
        self.set_data("input", data)
        self.run()
        self.finalize()
        return self.get_handle("output")
