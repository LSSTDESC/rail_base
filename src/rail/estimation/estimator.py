"""
Abstract base classes defining Estimators of individual galaxy redshift uncertainties.
"""

import gc
import sys
from typing import Any, Optional

import numpy as np
import qp
import tables_io

from rail.core.common_params import SHARED_PARAMS, SharedParams
from rail.core.data import (
    DataHandle,
    ModelHandle,
    ModelLike,
    QPHandle,
    TableHandle,
    TableLike,
)
from rail.core.enums import DistributionType
from rail.core.point_estimation import PointEstimationMixin
from rail.core.stage import RailStage

# for backwards compatibility, to avoid break stuff that imports it from here
from .informer import (  # pylint: disable=unused-import, relative-beyond-top-level
    CatInformer,
)


class CatEstimator(RailStage, PointEstimationMixin):
    """The base class for making photo-z posterior estimates from catalog-like inputs
    (i.e., tables with fluxes in photometric bands among the set of columns)

    Estimators use a generic "model", the details of which depends on the sub-class.

    Estimators take as "input" tabular data, apply the photo-z estimation and
    provide as "output" a ``QPEnsemble``, with per-object p(z).

    """

    name = "CatEstimator"
    entrypoint_function = "estimate"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(
        chunk_size=SharedParams.copy_param("chunk_size"),
        hdf5_groupname=SharedParams.copy_param("hdf5_groupname"),
        zmin=SharedParams.copy_param("zmin"),
        zmax=SharedParams.copy_param("zmax"),
        nzbins=SharedParams.copy_param("nzbins"),
        id_col=SharedParams.copy_param("id_col"),
        redshift_col=SharedParams.copy_param("redshift_col"),
        calc_summary_stats=SharedParams.copy_param("calc_summary_stats"),
    )

    config_options.update(
        **PointEstimationMixin.config_options.copy(),
    )
    inputs = [("model", ModelHandle), ("input", TableHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Estimator"""
        super().__init__(args, **kwargs)
        self._output_handle: QPHandle | None = None
        self.model = None
        self._partial_output = {}  # TODO: make this an ordered dict?

    def estimate(self, input_data: TableLike, **kwargs) -> QPHandle:
        """The main interface method for the photo-z estimation

        This will attach the input data (defined in ``inputs`` as "input") to this
        ``Estimator`` (for introspection and provenance tracking). Then call the
        ``run()``, ``validate()``, and ``finalize()`` methods.

        The run method will call ``_process_chunk()``, which needs to be implemented
        in the subclass, to process input data in batches. See ``RandomGaussEstimator``
        for a simple example.

        Finally, this will return a ``QPHandle`` for access to that output data.

        Parameters
        ----------
        input_data : TableLike
            A dictionary of all input data

        Returns
        -------
        QPHandle
            Handle providing access to QP ensemble with output data
        """
        self.set_data("input", input_data)
        self.validate()
        self.run()
        self.finalize()
        if len(self.outputs) == 1 or self.config.output_mode != "return":
            results = self.get_handle("output")
            if self.config.output_mode != "return":
                # only read from file if it wrote to a file
                results.read(force=True)
        # if there is more than one output and output_mode = return, return them all as a dictionary
        elif len(self.outputs) > 1 and self.config.output_mode == "return":
            results = {}
            for output in self.outputs:
                results[output[0]] = self.get_handle(output[0])
        return results

    def run(self) -> None:
        self.open_model(**self.config)

        iterator = self.input_iterator("input")
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            self._process_chunk(s, e, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._output_handle = None

    def _finalize_run(self) -> None:
        assert self._output_handle is not None
        if self.config.output_mode != "return":
            self._output_handle.finalize_write()
        elif self.config.output_mode == "return":
            # turn this into an ordered list by sorting the keys and then appending the
            # data into a sorted list
            # TODO: should we skip this bit and create a list from the start
            gathered_data = []
            start = 0
            for key in sorted(self._partial_output.keys()):
                gathered_data.append(self._partial_output[key])

            # set the output data handle path to None
            self._output_handle.path = None
            if len(gathered_data) <= 1:
                # if there is only one entry in the dictionary skip the concatenation
                self._output_handle.set_data(gathered_data[0])
            else:
                # concatenate all the chunks together
                gathered_ensembles = qp.concatenate(gathered_data)
                self._output_handle.set_data(gathered_ensembles)

            self._partial_output = {}  # clear the variable once we've used it

    def _process_chunk(
        self, start: int, end: int, data: TableLike, first: bool
    ) -> None:
        raise NotImplementedError(
            f"{self.name}._process_chunk is not implemented"
        )  # pragma: no cover

    @classmethod
    def default_distribution_type(cls) -> DistributionType:
        """Return the type of distribution that this estimator creates

        By default this is DistributionType.ad_hoc
        But this can be overridden by sub-classes to return
        DistributionType.posterior or DistributionType.likelihood if appropriate
        """
        return DistributionType.ad_hoc

    def _calculate_summary_stats(
        self,
        qp_dstn: qp.Ensemble,
    ) -> qp.Ensemble:

        if qp_dstn.ancil is None:  # pragma: no cover
            ancil_dict: dict[str, np.ndarray] = dict()
            qp_dstn.set_ancil(ancil_dict)

        quantiles = [0.025, 0.16, 0.5, 0.85, 0.975]
        quant_names = ["q2p5", "q16", "median", "q84", "97p5"]

        locs = qp_dstn.ppf(quantiles)
        for name_, vals_ in zip(quant_names, locs.T):
            qp_dstn.ancil[f"z_{name_}"] = np.expand_dims(vals_, -1)

        grid: np.ndarray | None = None

        if "z_mode" not in qp_dstn.ancil:
            grid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
            qp_dstn.ancil["z_mode"] = qp_dstn.mode(grid)

        try:
            qp_dstn.ancil["z_mean"] = qp_dstn.mean()
            qp_dstn.ancil["z_std"] = qp_dstn.std()
        except IndexError:  # pragma: no cover
            # this is needed b/c qp.MixMod pdf sometimes fails to compute moments
            grid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)
            pdfs = qp_dstn.pdf(grid)
            norms = pdfs.sum(axis=1)
            means = np.sum(pdfs * grid, axis=1) / norms
            diffs = (np.expand_dims(grid, -1) - means).T
            wt_diffs = diffs * diffs * pdfs
            stds = np.sqrt((wt_diffs).sum(axis=1) / norms)
            qp_dstn.ancil["z_mean"] = np.expand_dims(means, -1)
            qp_dstn.ancil["z_std"] = np.expand_dims(stds, -1)

        return qp_dstn

    def _do_chunk_output(
        self,
        qp_dstn: qp.Ensemble,
        start: int,
        end: int,
        first: bool,
        data: Optional[TableLike] = None,
    ) -> None:
        qp_dstn = self.calculate_point_estimates(qp_dstn)

        if self.config.calc_summary_stats:
            qp_dstn = self._calculate_summary_stats(qp_dstn)

        # if there is no ancil set by the calculate_point_estimate, initiate one
        if data is not None:
            if qp_dstn.ancil is None:  # pragma: no cover
                ancil_dict: dict[str, np.ndarray] = dict()
                qp_dstn.set_ancil(ancil_dict)
            assert qp_dstn.ancil is not None
            # if there is ID column in the input dataset, attach it to the ancil
            if self.config.id_col in data.keys():  # pragma: no cover
                qp_dstn.ancil.update(id=data[self.config.id_col])
            # if there is redshift column in the input dataset, attach it to the ancil
            if self.config.redshift_col in data.keys():  # pragma: no cover
                qp_dstn.ancil.update(redshift=data[self.config.redshift_col])

            if "distribution_type" not in qp_dstn.ancil:
                qp_dstn.ancil.update(
                    distribution_type=np.repeat(
                        self.default_distribution_type().value, end - start
                    )
                )

        if first:
            the_handle = self.add_handle("output", data=qp_dstn)
            assert isinstance(the_handle, QPHandle)
            self._output_handle = the_handle
            if self.config.output_mode != "return":
                self._output_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._output_handle is not None
        self._output_handle.set_data(qp_dstn, partial=True)
        if self.config.output_mode != "return":
            self._output_handle.write_chunk(start, end)
        elif self.config.output_mode == "return":
            self._partial_output[(start, end)] = qp_dstn
        return qp_dstn

    def _convert_table_format(
        self, data: TableLike, out_fmt_str: str = "numpyDict"
    ) -> TableLike:  # pragma: no cover
        """
        Utility function to convert existing Tabular data to a numpy dictionary,
        ingestable for most informer and estimators.
        To be called in _process_chunk().
        """
        # required format for informer/estimator
        out_fmt = tables_io.types.TABULAR_FORMAT_NAMES[out_fmt_str]
        out_data = tables_io.convert(data, out_fmt)
        # overwrite set_data
        return out_data


class PzEstimator(RailStage, PointEstimationMixin):
    """The base class for making photo-z posterior estimates from other pz inputs

    Estimators use a generic "model", the details of which depends on the sub-class.

    Estimators take as "input" a QPEnsemble, with other estimates and
    provide as "output" a ``QPEnsemble``, with per-object p(z).

    """

    name = "PzEstimator"
    entrypoint_function = "estimate"  # the user-facing science function for this class
    config_options = RailStage.config_options.copy()
    config_options.update(
        chunk_size=SharedParams.copy_param("chunk_size"),
        hdf5_groupname=SharedParams.copy_param("hdf5_groupname"),
    )
    config_options.update(
        **PointEstimationMixin.config_options.copy(),
    )
    inputs = [("model", ModelHandle), ("input", QPHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Estimator"""
        super().__init__(args, **kwargs)
        self._output_handle: QPHandle | None = None
        self.model = None

    def estimate(self, input_data: QPHandle, **kwargs) -> QPHandle:
        """The main interface method for the photo-z estimation

        This will attach the input data (defined in ``inputs`` as "input") to this
        ``Estimator`` (for introspection and provenance tracking). Then call the
        ``run()``, ``validate()``, and ``finalize()`` methods.

        The run method will call ``_process_chunk()``, which needs to be implemented
        in the subclass, to process input data in batches. See ``RandomGaussEstimator``
        for a simple example.

        Finally, this will return a ``QPHandle`` for access to that output data.

        Parameters
        ----------
        input_data : QPHandle
            A dictionary of all input data

        Returns
        -------
        QPHandle
            Handle providing access to QP ensemble with output data
        """
        self.set_data("input", input_data)
        self.validate()
        self.run()
        self.finalize()
        results = self.get_handle("output")
        results.read(force=True)
        return results

    def run(self) -> None:
        self.open_model(**self.config)

        iterator = self.input_iterator("input")
        first = True
        self._initialize_run()
        self._output_handle = None
        for s, e, test_data in iterator:
            print(f"Process {self.rank} running estimator on chunk {s:,} - {e:,}")
            sys.stdout.flush()
            self._process_chunk(s, e, test_data, first)
            first = False
            # Running garbage collection manually seems to be needed
            # to avoid memory growth for some estimators
            gc.collect()
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._output_handle = None

    def _finalize_run(self) -> None:
        assert self._output_handle is not None
        if self.config.output_mode != "return":
            self._output_handle.finalize_write()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:
        raise NotImplementedError(
            f"{self.name}._process_chunk is not implemented"
        )  # pragma: no cover

    def _do_chunk_output(
        self, qp_dstn: qp.Ensemble, start: int, end: int, first: bool
    ) -> None:
        qp_dstn = self.calculate_point_estimates(qp_dstn)
        if first:
            the_handle = self.add_handle("output", data=qp_dstn)
            assert isinstance(the_handle, QPHandle)
            self._output_handle = the_handle
            if self.config.output_mode != "return":
                self._output_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._output_handle is not None
        self._output_handle.set_data(qp_dstn, partial=True)
        if self.config.output_mode != "return":
            self._output_handle.write_chunk(start, end)
        return qp_dstn
