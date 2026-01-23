"""
Abstract base classes defining Summarizers of the redshift distribution of an ensemble of galaxies
"""

from typing import Any

import numpy as np
import qp

from rail.core.common_params import SHARED_PARAMS, SharedParams
from rail.core.data import DataHandle, ModelHandle, QPHandle, TableHandle, TableLike
from rail.core.stage import RailStage

# for backwards compatibility


class CatSummarizer(RailStage):  # pragma: no cover
    """The base class for classes that go from catalog-like tables
    to ensemble NZ estimates.

    CatSummarizer take as "input" a catalog-like table.  I.e., a
    table with fluxes in photometric bands among the set of columns.

    provide as "output" a QPEnsemble, with per-ensemble n(z).
    """

    name = "CatSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=SharedParams.copy_param("chunk_size"))
    inputs = [("input", TableHandle)]
    outputs = [("output", QPHandle)]

    def summarize(self, input_data: TableLike) -> QPHandle:
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
        input_data : TableLike
            Either a dictionary of all input data or a `TableHandle` providing access to the same

        Returns
        -------
        QPHandle
            Ensemble with n(z), and any ancillary data
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
    entrypoint_function = "summarize"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=SharedParams.copy_param("chunk_size"))
    inputs = [("model", ModelHandle), ("input", QPHandle)]
    outputs = [("output", QPHandle)]

    def summarize(self, input_data: qp.Ensemble, **kwargs) -> QPHandle:
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
        input_data : qp.Ensemble
            Per-galaxy p(z), and any ancillary data associated with it

        Returns
        -------
        QPHandle
            Ensemble with n(z), and any ancillary data
        """
        self.set_data("input", input_data)
        self.run()
        self.finalize()
        if len(self.outputs) == 1 or self.config.output_mode != "return":
            results = self.get_handle("output")
        # if there is more than one output and output_mode = return, return them all as a dictionary
        elif len(self.outputs) > 1 and self.config.output_mode == "return":
            results = {}
            for output in self.outputs:
                results[output[0]] = self.get_handle(output[0])
        return results

        # return self.get_handle("output")

    def _broadcast_bootstrap_matrix(self) -> np.ndarray | None:
        rng = np.random.default_rng(seed=self.config.seed)
        # Only one of the nodes needs to produce the bootstrap indices
        ngal = self._input_length
        if self.rank == 0:
            bootstrap_matrix = rng.integers(
                low=0, high=ngal, size=(ngal, self.config.n_samples)
            )
        else:  # pragma: no cover
            bootstrap_matrix = None
        if self.comm is not None:  # pragma: no cover
            self.comm.Barrier()
            bootstrap_matrix = self.comm.bcast(bootstrap_matrix, root=0)
        return bootstrap_matrix

    def _join_histograms(
        self, bvals: np.ndarray, yvals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        bvals_r = self.comm.reduce(bvals)
        yvals_r = self.comm.reduce(yvals)
        return (bvals_r, yvals_r)


class SZPZSummarizer(RailStage):  # pragma: no cover
    """The base class for classes that use two sets of data: a photometry sample with
    spec-z values, and a photometry sample with unknown redshifts, e.g. minisom_som and
    outputs a QP Ensemble with bootstrap realization of the N(z) distribution
    """

    name = "SZPZtoNZSummarizer"
    entrypoint_function = "summarize"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(chunk_size=SharedParams.copy_param("chunk_size"))
    inputs = [
        ("input", TableHandle),
        ("spec_input", TableHandle),
        ("model", ModelHandle),
    ]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Estimator that can sample galaxy data."""
        super().__init__(args, **kwargs)
        self.model = None
        # NOTE: open model removed from init, need to put an
        # `open_model` call explicitly in the run method for
        # each summarizer.

    def summarize(
        self, input_data: qp.Ensemble, spec_data: np.ndarray, **kwargs
    ) -> qp.Ensemble:
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
        input_data : qp.Ensemble
            Per-galaxy p(z), and any ancillary data associated with it
        spec_data : np.ndarray
            Spectroscopic data

        Returns
        -------
        qp.Ensemble
            Ensemble with n(z), and any ancillary data
        """
        self.set_data("input", input_data)
        self.set_data("spec_input", spec_data)
        self.run()
        self.finalize()
        if len(self.outputs) == 1 or self.config.output_mode != "return":
            return self.get_handle("output")
        # if there is more than one output and output_mode = return, return them all as a dictionary
        elif len(self.outputs) > 1 and self.config.output_mode == "return":
            results = {}
            for output in self.outputs:
                results[output[0]] = self.get_handle(output[0])
            return results
