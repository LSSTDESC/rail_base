"""Abstract base class defining a noisifier.

The key feature here is the run adds noise to the catalog.
Intended subclasses are noisifier that adds LSST noise / other telescope noise
"""

from ceci.config import StageParameter as Param

from rail.core.data import PqHandle, TableLike
from rail.core.stage import RailStage


class Noisifier(RailStage):
    """Base class Noisifier, which adds noise to the input catalog

    Noisifier take "input" data in the form of pandas dataframes in Parquet
    files and provide as "output" another pandas dataframes written to Parquet
    files.
    """

    name = "Noisifier"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(
        seed=Param(
            default=None,
            required=False,
            msg="Set to an `int` to force reproducible results.",
        ),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def _initNoiseModel(self) -> None:  # pragma: no cover
        raise NotImplementedError("Noisifier._initNoiseModel()")

    def _addNoise(self) -> None:  # pragma: no cover
        raise NotImplementedError("Noisifier._addNoise()")

    def __call__(
        self, sample: TableLike, seed: int | None = None, **kwargs
    ) -> PqHandle:
        """
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

        Parameters
        ----------
        sample : TableLike
            The sample to be degraded.

        seed : int, optional
            An integer to set the numpy random seed, by default None.

        Returns
        -------
        PqHandle
            A handle giving access to a table with degraded sample.
        """
        if seed is not None:
            self.config.seed = seed
        self.set_data("input", sample)

        self.run()
        self.finalize()
        return self.get_handle("output")

    def run(self) -> None:
        self._initNoiseModel()
        self._addNoise()
