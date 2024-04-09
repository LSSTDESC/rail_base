"""Abstract base class defining a selector.

The key feature here is make selection to either the photometric or spectroscopic catalog. 
Intended subclasses spectroscopic selection, probability selection on a grid for the photometry, 
or pure photometric selection. 
"""

from rail.core.stage import RailStage
from rail.core.data import PqHandle


class Selector(RailStage):
    """Base class Selector, which makes selection to the catalog

    Selector take "input" data in the form of pandas dataframes in Parquet 
    files and provide as "output" another pandas dataframes written to Parquet 
    files.
    """

    name = 'Selector'
    config_options = RailStage.config_options.copy()
    inputs = [('input', PqHandle)]
    outputs = [('output', PqHandle)]

    def __init__(self, args, comm=None):
        """Initialize Noisifier that can add noise to photometric data"""
        RailStage.__init__(self, args, comm=comm)

    def __call__(self, sample):
        """The main interface method for ``Selector``.

        Adds noise to the input catalog

        This will attach the input to this `Selector` 

        Then it will call the select() which add a flag column to the catalog. flag=1 means 
        selected, 0 means dropped.

        If dropRows = True, the dropped rows will not be presented in the output catalog, 
        otherwise, all rows will be presented. 

        Finally, this will return a PqHandle providing access to that output 
        data.

        Parameters
        ----------
        sample : table-like
            The sample to be selected

        Returns
        -------
        output_data : PqHandle
            A handle giving access to a table with selected sample
        """
        
        self.run()
        
        return self.get_handle('output')

    def run(self):
        
        self.set_data('input', sample)
        
        self.select()
        
        self.finalize(dropRows = self.args['dropRows'])