from typing import Any

def euclid_deep_error_model(**kwargs) -> Any:
    """
    The Euclid Deep Error model, defined by peEuclidDeepErrorParams and
    peEuclidDeepErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.EuclidDeepErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def euclid_error_model(**kwargs) -> Any:
    """
    The Euclid Error model, defined by peEuclidErrorParams and peEuclidErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.EuclidErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def euclid_wide_error_model(**kwargs) -> Any:
    """
    The Euclid Wide Error model, defined by peEuclidWideErrorParams and
    peEuclidWideErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.EuclidWideErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def lsst_error_model(**kwargs) -> Any:
    """
    The LSST Error model, defined by peLsstErrorParams and peLsstErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.LSSTErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def photo_error_model(**kwargs) -> Any:
    """
    The Base Model for photometric errors.

    This is a wrapper around the error model from PhotErr. The parameter
    docstring below is dynamically added by the installed version of PhotErr:

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.PhotoErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def roman_deep_error_model(**kwargs) -> Any:
    """
    The Roman Deep Error model, defined by peRomanDeepErrorParams and
    peRomanDeepErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.RomanDeepErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def roman_error_model(**kwargs) -> Any:
    """
    The Roman Error model, defined by peRomanErrorParams and peRomanErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.RomanErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def roman_medium_error_model(**kwargs) -> Any:
    """
    The Roman Medium Error model, defined by peRomanMediumErrorParams and
    peRomanMediumErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.RomanMediumErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def roman_ultra_deep_error_model(**kwargs) -> Any:
    """
    The Roman UltraDeep Error model, defined by peRomanUltraDeepErrorParams and
    peRomanUltraDeepErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.RomanUltraDeepErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """

def roman_wide_error_model(**kwargs) -> Any:
    """
    The Roman WideError model, defined by peRomanWideErrorParams and
    peRomanWideErrorModel

    ---

    The main interface method for ``Noisifier``.

    Adds noise to the input catalog

    This will attach the input to this `Noisifier`

    Then it will call the _initNoiseModel() and _addNoise(), which need to be
    implemented by the sub-classes.

    The _initNoiseModel() method will initialize the noise model of the sub-classes, and
    store the noise model as self.noiseModel

    The _addNoise() method will add noise to the flux and magnitude of the column of the
    catalog.

    The finalize() method will check the end results (like preserving number of rows)

    Finally, this will return a PqHandle providing access to that output
    data.

    ---

    This function was generated from the function
    rail.creation.degraders.photometric_errors.RomanWideErrorModel.__call__

    Parameters
    ----------
    sample : TableLike, required
        The sample to be degraded.
    seed : int, optional
        Set to an `int` to force reproducible results.
        Default: None
        An integer to set the numpy random seed, by default None.

    Returns
    -------
    pandas.core.frame.DataFrame
        A handle giving access to a table with degraded sample.
    """
