"""
Abstract base classes defining Summarizers of the redshift distribution of an ensemble of galaxies
"""

from typing import Any, Generator

import numpy as np
import qp

from rail.core.common_params import SHARED_PARAMS, SharedParams, TOMOGRAPHY_ALL, TOMOGRAPHY_NONE
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

    def summarize(self, input_data: qp.Ensemble, **kwargs) ->  QPHandle | dict:
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
        return self._do_return()

    def _do_return(self) -> QPHandle | dict:
        if len(self.outputs) == 1 or self.config.output_mode != "return":
            results = self.get_handle("output")
        # if there is more than one output and output_mode = return, return them all as a dictionary
        elif len(self.outputs) > 1 and self.config.output_mode == "return":
            results = {}
            for output in self.outputs:
                results[output[0]] = self.get_handle(output[0])
        return results
    
    def _get_n_tomo_bins(self) -> int:
        if 'n_tomo_bins' in self.config:
            return self.config.n_tomo_bins
        else:
            return 1
    
    def _setup_iterator(self) -> Generator:
        """ Set the iterator to interate over chunks, 
        and to optionally apply masking
        """

        if 'selected_bin' in self.config:
            selected_bin = self.config.selected_bin
        else:
            selected_bin = TOMOGRAPHY_NONE

        if 'tomography_bins' in self.config:
            if self.config.tomography_bins in ["none", None]:
                selected_bin = TOMOGRAPHY_NONE
        else:
            selected_bin = TOMOGRAPHY_NONE

        if 'n_tomo_bins' in self.config:
            n_tomo_bins = self.config.n_tomo_bins
        else:
            n_tomo_bins = 1
                
        if selected_bin == TOMOGRAPHY_NONE:
            itrs = [self.input_iterator("input")]
        else:
            itrs = [
                self.input_iterator("input"),
                self.input_iterator("tomography_bins"),
            ]

        for it in zip(*itrs):
            first = True
            mask = None
            for s, e, d in it:
                if first:
                    start = s
                    end = e
                    pz_data = d
                    first = False
                else:
                    try:
                        bin_assignments = d['class_id']
                    except KeyError:
                        bin_assignments = d['tomo_bin_index']
                    if n_tomo_bins > 1:
                        all_masks = []
                        for i in range(selected_bin, selected_bin+n_tomo_bins):
                            all_masks.append(bin_assignments == i)
                        mask = np.vstack(all_masks)
                    elif selected_bin == TOMOGRAPHY_ALL:
                        mask = bin_assignments >= 0
                    else:
                        mask = bin_assignments == selected_bin
            if mask is None:
                mask = np.ones(
                    pz_data.npdf,  # pylint: disable=possibly-used-before-assignment
                    dtype=bool,
                )
            yield start, end, pz_data, mask  # pylint: disable=possibly-used-before-assignment
    
    def _join_histograms(
        self, bvals: np.ndarray, yvals: np.ndarray, n_objects: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        bvals_r = self.comm.reduce(bvals)
        yvals_r = self.comm.reduce(yvals)
        n_objects_r = self.comm.reduce(n_objects)
        return (bvals_r, yvals_r, n_objects_r)


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
