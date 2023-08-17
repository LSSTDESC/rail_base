"""
Stage to produce point estimates from qp.Ensembles of PDFs after they are produced
"""

class Reducer(RailStage):
    """ A stage for producing point estimates from existing PDFs"""

    name = 'Reducer'
    config_options = RailStage.config_options.copy()
    config_options.update(
    #     chunk_size=10000,
    #     hdf5_groupname=SHARED_PARAMS['hdf5_groupname'],
        calculated_point_estimates=SHARED_PARAMS['calculated_point_estimates'])
    inputs = [('input', QPHandle)]
    outputs = [('output', TableHandle)]

    def __init__(self, args, comm=None):
        """
        Initialize the reducer as a RailStage object
        """
        RailStage.__init__(self, args, comm=comm)
        self._output_handle = None
    
    def reduce(self, data, calculated_point_estimates=[]):
        """
        Produce point estimates from existing PDFs

        This will attach the input data and truth to the Reducer
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to the Reducer
        by using `self.add_data('output', output_data)`.

        Parameters
        ----------
        data : qp.Ensemble
            The PDFs to reduce
    
        Returns
        -------
        output : Table-like
            The point estimates
        """
        self.set_data('input', data)
        self.run()
        self.finalize()
        return self.get_handle('output')
    
    def run(self):
        """
        Perform the desired point estimation reductions from PDFs

        Notes
        -----
        Copy this from Drew's ongoing work on #42!
        """
        raise NotImplementedError(f"{self.name}.run is not implemented")