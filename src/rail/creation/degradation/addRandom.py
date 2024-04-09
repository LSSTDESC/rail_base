"""Add a column of random numbers to a dataframe."""

import numpy as np

from ceci.config import StageParameter as Param
from rail.creation.noisifier import Noisifier

class AddColumnOfRandom(Noisifier):
    """Add a column of random numbers to a dataframe
    """

    name = "AddColumnOfRandom"
    config_options = Noisifier.config_options.copy()
    config_options.update(
        col_name=Param(str, "chaos_bunny", msg="Name of the column with random numbers"),
    )

    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Noisifier initialization
        """
        Noisifier.__init__(self, args, comm=comm)

    def _initNoiseModel(self):  # pragma: no cover
        np.random.seed(self.config.seed)
        
    def _addNoise(self):  # pragma: no cover        
        data = self.get_data('input')
        copy = data.copy()
        copy.insert(0, self.config.col_name, np.random.uniform(size=len(copy)))
        self.add_data("output", copy)
