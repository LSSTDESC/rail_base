"""Abstract base class defining a noisifier.

The key feature here is the run adds noise to the catalog. 
Intended subclasses are noisifier that adds LSST noise / other telescope noise
"""

from rail.core.stage import RailStage
from rail.core.data import PqHandle


class Noisifier(RailStage):
    """Base class Noisifier, which adds noise to the input catalog

    Noisifier take "input" data in the form of pandas dataframes in Parquet 
    files and provide as "output" another pandas dataframes written to Parquet 
    files.
    """

    name = 'Noisifier'
    config_options = RailStage.config_options.copy()
    config_options.update(seed=1337)
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        """Initialize Noisifier that can add noise to photometric data"""
        RailStage.__init__(self, args, comm=comm)

    def __call__(self, sample, seed: int = None):
        """The main interface method for ``Noisifier``.

        Adds noise to the input catalog

        This will attach the input to this `Noisifier` 

        Then it will call the initNoiseModel() and addNoise(), which need to be
        implemented by the sub-classes.

        The initNoiseModel() method will initialize the noise model of the sub-classes, and 
        store the noise model as self.noiseModel
        
        The addNoise() method will add noise to the flux and magnitude of the column of the
        catalog. 
        
        The finalize() method will check the end results (like preserving number of rows)

        Finally, this will return a PqHandle providing access to that output 
        data.

        Parameters
        ----------
        sample : table-like
            The sample to be degraded
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        output_data : PqHandle
            A handle giving access to a table with degraded sample
        """
        if seed is not None:
            self.config.seed = seed
        self.set_data('input', sample)
        
        self.initNoiseModel()
        self.addNoise(self.noiseModel)
        
        self.finalize()
        
        return self.get_handle('output')
