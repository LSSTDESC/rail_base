from typing import Any

def column_mapper(**kwargs) -> Any:
    """
    Utility stage that remaps the names of columns.

    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data.rename(columns=self.config.columns,
    in_place=self.config.in_place)`

    ---

    Return a table with the columns names changed

    ---

    This function was generated from the function
    rail.tools.table_tools.ColumnMapper.__call__

    Parameters
    ----------
    input : TableLike
        The data to be renamed
    columns : dict
        Map of columns to rename
    in_place : bool, optional
        Update file in place
        Default: False

    Returns
    -------
    pandas.core.frame.DataFrame
        The degraded sample
    """

def row_selector(**kwargs) -> Any:
    """
    Utility Stage that sub-selects rows from a table by index

    1. This operates on pandas dataframs in parquet files.

    2. In short, this does:
    `output_data = input_data[self.config.start_row:self.config.stop_row]`

    ---

    Return a table with the columns names changed

    ---

    This function was generated from the function
    rail.tools.table_tools.RowSelector.__call__

    Parameters
    ----------
    input : TableLike
        The data to be renamed
    start_row : int
        starting row number
    stop_row : int
        Stoppig row number

    Returns
    -------
    pandas.core.frame.DataFrame
        The degraded sample
    """

def table_converter(**kwargs) -> Any:
    """
    Utility stage that converts tables from one format to anothe

    FIXME, this is hardwired to convert parquet tables to Hdf5Tables.
    It would be nice to have more options here.

    ---

    Return a converted table

    ---

    This function was generated from the function
    rail.tools.table_tools.TableConverter.__call__

    Parameters
    ----------
    input : TableLike
        The data to be converted
    output_format : str
        Format of output table

    Returns
    -------
    dict
        The converted version of the table
    """
