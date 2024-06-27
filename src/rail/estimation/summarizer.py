"""
Abstract base classes defining Summarizers of the redshift distribution of an ensemble of galaxies
"""

import numpy as np

from rail.core.data import QPHandle, TableHandle, ModelHandle
from rail.core.stage import RailStage

# for backwards compatibility


class CatSummarizer(RailStage):
    """The base class for classes that go from catalog-like tables
    to ensemble NZ estimates.

    CatSummarizer take as "input" a catalog-like table.  I.e., a
    table with fluxes in photometric bands among the set of columns.

    provide as "output" a QPEnsemble, with per-ensemble n(z).
    """

    name = "CatSummarizer"
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [("input", TableHandle)]
    outputs = [("output", QPHandle)]

    def summarize(self, input_data):
        """The main run method for the summarization, should be implemented
        in the specific subclass.

        This will attach the input_data to this `CatSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this `CatSummarizer`
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `dict` or `ModelHandle`
            Either a dictionary of all input data or a `ModelHandle` providing access to the same

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")


class PZSummarizer(RailStage):
    """The base class for classes that go from per-galaxy PZ estimates to ensemble NZ estimates

    PZSummarizer take as "input" a `qp.Ensemble` with per-galaxy PDFs, and
    provide as "output" a QPEnsemble, with per-ensemble n(z).
    """

    name = "PZtoNZSummarizer"
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [("model", ModelHandle), ("input", QPHandle)]
    outputs = [("output", QPHandle)]

    def summarize(self, input_data):
        """The main run method for the summarization, should be implemented
        in the specific subclass.

        This will attach the input_data to this `PZtoNZSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _broadcast_bootstrap_matrix(self):        
        rng = np.random.default_rng(seed=self.config.seed)
        # Only one of the nodes needs to produce the bootstrap indices
        ngal = self._input_length
        if self.rank == 0:
            bootstrap_matrix = rng.integers(
                low=0, high=ngal, size=(ngal, self.config.nsamples)
            )
        else:  # pragma: no cover
            bootstrap_matrix = None
        if self.comm is not None:  # pragma: no cover
            self.comm.Barrier()
            bootstrap_matrix = self.comm.bcast(bootstrap_matrix, root=0)
        return bootstrap_matrix

    def _join_histograms(self, bvals, yvals):  # pragma: no cover
        bvals_r = self.comm.reduce(bvals)
        yvals_r = self.comm.reduce(yvals)
        return (bvals_r, yvals_r)


class SZPZSummarizer(RailStage):
    """The base class for classes that use two sets of data: a photometry sample with
    spec-z values, and a photometry sample with unknown redshifts, e.g. minisom_som and
    outputs a QP Ensemble with bootstrap realization of the N(z) distribution
    """

    name = "SZPZtoNZSummarizer"
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=10000)
    inputs = [
        ("input", TableHandle),
        ("spec_input", TableHandle),
        ("model", ModelHandle),
    ]
    outputs = [("output", QPHandle)]

    def __init__(self, args, **kwargs):
        """Initialize Estimator that can sample galaxy data."""
        super().__init__(args, **kwargs)
        self.model = None
        # NOTE: open model removed from init, need to put an
        # `open_model` call explicitly in the run method for
        # each summarizer.

    def open_model(self, **kwargs):
        """Load the mode and/or attach it to this Summarizer

        Notes
        -----
        Keywords are potentially used to build the model.
        By default we just use 'model'  which can be an `object`, `str` or `ModelHandle`

        I.e., either an object with a trained model, a path pointing to a file that
        can be read to obtain the trained model, or a `ModelHandle` providing access to the trained model.

        Returns
        -------
        self.model : `object`
            The object encapsulating the trained model.
        """
        model = kwargs.get("model", None)
        if model is None or model == "None":  # pragma: no cover
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

    def summarize(self, input_data, spec_data):
        """The main run method for the summarization, should be implemented
        in the specific subclass.

        This will attach the input_data to this `SZandPhottoNZSummarizer`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Finally, this will return a QPHandle providing access to that output data.

        Parameters
        ----------
        input_data : `qp.Ensemble`
            Per-galaxy p(z), and any ancilary data associated with it

        Returns
        -------
        output: `qp.Ensemble`
            Ensemble with n(z), and any ancilary data
        """
        self.set_data("input", input_data)
        self.set_data("spec_input", spec_data)
        self.run()
        self.finalize()
        return self.get_handle("output")
