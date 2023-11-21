"""
Abstract base classes defining classifiers.
"""

import gc
from rail.core.data import QPHandle, TableHandle, ModelHandle, Hdf5Handle
from rail.core.stage import RailStage


class CatClassifier(RailStage):  # pragma: no cover
    """The base class for assigning classes to catalogue-like table.

    Classifier uses a generic "model", the details of which depends on the sub-class.

    CatClassifier take as "input" a catalogue-like table, assign each object into
    a tomographic bin, and provide as "output" a tabular data which can be appended
    to the catalogue.
    """

    name = "CatClassifier"
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000, hdf5_groupname=str)
    inputs = [("model", ModelHandle), ("input", TableHandle)]
    outputs = [("output", TableHandle)]

    def __init__(self, args, comm=None):
        """Initialize Classifier"""
        RailStage.__init__(self, args, comm=comm)
        self._output_handle = None
        self.model = None
        if not isinstance(args, dict):  # pragma: no cover
            args = vars(args)
        self.open_model(**args)

    def open_model(self, **kwargs):
        """Load the model and/or attach it to this Classifier

        Parameters
        ----------
        model : `object`, `str` or `ModelHandle`
            Either an object with a trained model,
            a path pointing to a file that can be read to obtain the trained model,
            or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """
        model = kwargs.get("model", None)
        if model is None or model == "None":
            self.model = None
            return self.model
        if isinstance(model, str):
            self.model = self.set_data("model", data=None, path=model)
            self.config["model"] = model
            return self.model
        if isinstance(model, ModelHandle):
            if model.has_path:
                self.config["model"] = model.path
        self.model = self.set_data("model", model)
        return self.model

    def classify(self, input_data):
        """The main run method for the classifier, should be implemented
        in the specific subclass.

        This will attach the input_data to this `CatClassifier`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Classifier
        by using `self.add_data('output', output_data)`.

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        input_data : `dict`
            A dictionary of all input data

        Returns
        -------
        output: `dict`
            Class assignment for each galaxy.
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")


class PZClassifier(RailStage):
    """The base class for assigning classes (tomographic bins) to per-galaxy PZ 
    estimates.

    PZClassifier takes as "input" a `qp.Ensemble` with per-galaxy PDFs, and
    provides as "output" tabular data which can be appended to the catalogue.
    """
    name='PZClassifier'
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [('input', QPHandle)]
    outputs = [('output', Hdf5Handle)]
    
    def __init__(self, args, comm=None):
        """Initialize the PZClassifier.

        Parameters
        ----------
        args : dict
            Configuration arguments for the classifier.
        comm : MPI.Comm, optional
            MPI communicator for parallel processing.
        """
        RailStage.__init__(self, args, comm=comm)
        self._output_handle = None
        

    def classify(self, input_data):
        """The main run method for the classifier, should be implemented
        in the specific subclass.

        This will attach the input_data to this `PZClassifier`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this 
        Classifier by using `self.add_data('output', output_data)`.

        The run() method relies on the _process_chunk() method, which should be 
        implemented by subclasses to perform the actual classification on each 
        chunk of data. The results from each chunk are then combined in the 
        _finalize_run() method. (Alternatively, override run() in a subclass to 
        perform the classification without parallelization.)

        Finally, this will return a TableHandle providing access to that output data.

        Parameters
        ----------
        input_data : `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        output: `dict`
            Class assignment for each galaxy, typically in the form of a 
            dictionary with IDs and class labels.
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle('output')
    

    def _finalize_run(self):
        """Finalize the classification process after processing all chunks."""
        self._output_handle.finalize_write()
    
    def _process_chunk(self, start, end, data, first):
        """Process a chunk of data.

        This method should be implemented in subclasses to perform the actual
        classification on a chunk of data.

        Parameters
        ----------
        start : int
            The starting index of the chunk.
        end : int
            The ending index of the chunk.
        data : qp.Ensemble
            The data chunk to be processed.
        first : bool
            True if this is the first chunk, False otherwise.
        """
        raise NotImplementedError(f"{self.name}._process_chunk is not implemented")  # pragma: no cover
    
    def _do_chunk_output(self, class_id, start, end, first):
        """Handle the output of a processed chunk.

        Parameters
        ----------
        class_id : dict
            The classification results for the chunk.
        start : int
            The starting index of the chunk.
        end : int
            The ending index of the chunk.
        first : bool
            True if this is the first chunk, False otherwise.
        """
        if first:
            self._output_handle = self.add_handle('output', data=class_id)
            self._output_handle.initialize_write(self._input_length, communicator=self.comm)
        self._output_handle.set_data(class_id, partial=True)
        self._output_handle.write_chunk(start, end)

    def run(self):
        """Processes the input data in chunks and performs classification.

        This method iterates over chunks of the input data, calling the 
        _process_chunk method for each chunk to perform the actual classification. 

        The _process_chunk method should be implemented by subclasses to define 
        the specific classification logic.
        """
        test_data = self.get_data('input')
        
        iterator = self.input_iterator('input')
        first = True
        
        for start, end, test_data in iterator:
            #print(f"Process {self.rank} running estimator on chunk {start} - {end}")
            self._process_chunk(start, end, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()
